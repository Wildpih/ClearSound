"""Track Sound Donor Extractor — extracts clean, un-muffled sound events
from non-vocal sections of the source audio to use as authentic replacement
donors for muffled sounds in vocal gaps (e.g., alarm beeps, phone rings,
foley, clicks, footsteps).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
from rich.console import Console

from clearsound.core.audio_utils import (
    compute_crest_factor,
    compute_dominant_frequency,
    compute_mfcc,
    compute_spectral_flatness,
    format_time,
)
from clearsound.library.sound_library import SoundEntry

console = Console()


@dataclass
class DonorSound:
    """A clean sound event extracted from the non-vocal sections of the audio."""

    path: Path
    name: str
    start_sec: float
    end_sec: float
    duration_sec: float
    dominant_freq: float
    peak_ratio: float
    tonality: float
    crest_factor: float
    is_tonal: bool
    is_transient: bool
    audio: np.ndarray
    mfcc: np.ndarray

    def to_sound_entry(self) -> SoundEntry:
        """Convert to a SoundEntry compatible with the sound library and matcher."""
        tag = "писк/сигнал" if self.is_tonal else "шум/фоли"
        entry_name = f"🎯 Донор: {tag} ({format_time(self.start_sec)}, {self.dominant_freq:.0f} Гц)"
        return SoundEntry(
            path=self.path,
            name=entry_name,
            category="donor",
            duration_sec=self.duration_sec,
            tags=["donor", "self_sample", f"{int(self.dominant_freq)}hz", "tonal" if self.is_tonal else "transient"],
        )


class DonorExtractor:
    """Extracts authentic clean sound donors from non-vocal sections of the audio track."""

    def __init__(self, temp_dir: Path):
        self.temp_dir = Path(temp_dir) / "donors"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def extract_donors(
        self,
        no_vocals: np.ndarray,
        vocal_regions: List[Tuple[int, int]],
        sr: int = 44100,
        min_duration_sec: float = 0.05,
        max_duration_sec: float = 2.5,
    ) -> List[DonorSound]:
        """Scan non-vocal sections of no_vocals audio to discover clean donor events."""
        console.print("\n[bold cyan]🎯 Сканируем аудиодорожку на чистые звуки-доноры...[/]")

        audio_mono = np.mean(no_vocals, axis=0) if no_vocals.ndim > 1 else no_vocals
        total_samples = len(audio_mono)

        # ── 1. Determine clean (non-vocal) intervals ──────────────────
        clean_intervals: List[Tuple[int, int]] = []
        cur_pos = 0

        # Sort vocal regions
        sorted_vocs = sorted(vocal_regions, key=lambda r: r[0])
        # Margin around voice to avoid trailing phonemes
        margin_samples = int(0.12 * sr)

        for v_st, v_en in sorted_vocs:
            clean_end = max(0, v_st - margin_samples)
            if clean_end - cur_pos >= int(min_duration_sec * sr):
                clean_intervals.append((cur_pos, clean_end))
            cur_pos = min(total_samples, v_en + margin_samples)

        if cur_pos < total_samples:
            clean_intervals.append((cur_pos, total_samples))

        if not clean_intervals:
            console.print("  [yellow]Нет свободных от голоса интервалов для поиска доноров.[/]")
            return []

        # ── 2. Discover energy bursts & events in clean intervals ─────
        candidates: List[DonorSound] = []
        win_samples = int(0.08 * sr)
        hop_samples = int(0.02 * sr)

        for c_st, c_en in clean_intervals:
            chunk = audio_mono[c_st:c_en]
            if len(chunk) < win_samples:
                continue

            # Compute short-time energy
            num_frames = (len(chunk) - win_samples) // hop_samples
            if num_frames <= 0:
                continue

            energies = np.array([
                np.sqrt(np.mean(chunk[i * hop_samples : i * hop_samples + win_samples] ** 2))
                for i in range(num_frames)
            ])

            base_energy = np.median(energies)
            std_energy = np.std(energies)
            threshold = base_energy + 1.2 * std_energy

            # Ignore absolute silence
            if threshold < 1e-4:
                continue

            active = energies > threshold
            in_event = False
            ev_start = 0

            for frame_idx, is_act in enumerate(active):
                if is_act and not in_event:
                    in_event = True
                    ev_start = frame_idx * hop_samples
                elif not is_act and in_event:
                    in_event = False
                    ev_end = min(len(chunk), frame_idx * hop_samples + win_samples)
                    dur = (ev_end - ev_start) / sr
                    if min_duration_sec <= dur <= max_duration_sec:
                        ev_audio = chunk[ev_start:ev_end]
                        donor = self._analyze_candidate(ev_audio, c_st + ev_start, c_st + ev_end, sr)
                        if donor is not None:
                            candidates.append(donor)

            if in_event:
                ev_end = len(chunk)
                dur = (ev_end - ev_start) / sr
                if min_duration_sec <= dur <= max_duration_sec:
                    ev_audio = chunk[ev_start:ev_end]
                    donor = self._analyze_candidate(ev_audio, c_st + ev_start, c_st + ev_end, sr)
                    if donor is not None:
                        candidates.append(donor)

        # ── 3. Cluster and deduplicate similar donors ─────────────────
        unique_donors = self._deduplicate_donors(candidates)
        console.print(f"  [bold green]✓ Извлечено чистых звуков-доноров из трека: {len(unique_donors)}[/]")
        for d in unique_donors[:5]:
            kind = "Писк/Тон" if d.is_tonal else ("Импульс" if d.is_transient else "Звук")
            console.print(f"    • {kind} на {format_time(d.start_sec)} ({d.duration_sec:.2f}с, {d.dominant_freq:.0f} Гц)")

        return unique_donors

    def _analyze_candidate(self, audio_slice: np.ndarray, start_samp: int, end_samp: int, sr: int) -> Optional[DonorSound]:
        """Analyze acoustic features of an audio slice and save it if valid."""
        if len(audio_slice) < 256:
            return None

        # Energy check
        rms = np.sqrt(np.mean(audio_slice ** 2))
        if rms < 0.005:  # too quiet
            return None

        dom_freq, peak_ratio = compute_dominant_frequency(audio_slice, sr, min_freq=180.0)
        flatness = compute_spectral_flatness(audio_slice)
        tonality = float(np.clip(1.0 - flatness, 0.0, 1.0))
        crest = compute_crest_factor(audio_slice)
        dur = len(audio_slice) / sr

        is_tonal = bool(tonality > 0.35 and peak_ratio >= 3.0 and dom_freq >= 200.0)
        is_transient = bool(crest > 3.0 and dur <= 0.6)

        # Only keep interesting events (tonal beeps/rings or sharp transients/clicks)
        if not is_tonal and not is_transient:
            return None

        try:
            mfcc_mat = compute_mfcc(audio_slice, sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc_mat, axis=1)
        except Exception:
            mfcc_mean = np.zeros(13, dtype=np.float32)

        start_sec = start_samp / sr
        end_sec = end_samp / sr
        tag = f"{int(dom_freq)}hz" if is_tonal else "transient"
        wav_filename = f"donor_{int(start_sec * 1000)}ms_{tag}.wav"
        wav_path = self.temp_dir / wav_filename

        try:
            # Normalize to clean audible level
            peak_val = np.max(np.abs(audio_slice))
            norm_audio = audio_slice / (peak_val + 1e-8) * 0.75
            sf.write(str(wav_path), norm_audio, sr)
        except Exception as e:
            console.print(f"[dim red]Не удалось сохранить донор {wav_filename}: {e}[/]")
            return None

        return DonorSound(
            path=wav_path,
            name=f"Донор {format_time(start_sec)} ({dom_freq:.0f} Гц)",
            start_sec=start_sec,
            end_sec=end_sec,
            duration_sec=dur,
            dominant_freq=dom_freq,
            peak_ratio=peak_ratio,
            tonality=tonality,
            crest_factor=crest,
            is_tonal=is_tonal,
            is_transient=is_transient,
            audio=audio_slice,
            mfcc=mfcc_mean,
        )

    def _deduplicate_donors(self, donors: List[DonorSound], max_donors: int = 15) -> List[DonorSound]:
        """Keep distinct representative donors, removing near-identical repeats."""
        if not donors:
            return []

        # Sort by clarity / peak ratio (highest quality first)
        donors.sort(key=lambda d: (-d.peak_ratio, -d.tonality))

        selected: List[DonorSound] = []
        for d in donors:
            # Check if we already have a donor with very close dominant frequency and duration
            duplicate = False
            for s in selected:
                freq_diff = abs(d.dominant_freq - s.dominant_freq) / max(1.0, s.dominant_freq)
                if d.is_tonal and s.is_tonal and freq_diff < 0.08:
                    duplicate = True
                    break
                if d.is_transient and s.is_transient and abs(d.duration_sec - s.duration_sec) < 0.08:
                    # same kind of transient close in time
                    if abs(d.start_sec - s.start_sec) < 0.8:
                        duplicate = True
                        break

            if not duplicate:
                selected.append(d)
                if len(selected) >= max_donors:
                    break

        # Re-sort chronologically for user convenience
        selected.sort(key=lambda d: d.start_sec)
        return selected
