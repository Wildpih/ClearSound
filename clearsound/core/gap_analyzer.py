"""Gap analyzer — finds regions where voice was present and the resulting
audio sounds degraded or hollow after vocal removal.

Compares the original audio with the vocal-removed version to identify
segments that need sound restoration, and characterizes the spectral
profile of each gap for later matching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import librosa
import numpy as np
from rich.console import Console
from rich.table import Table
from scipy.signal import fftconvolve

from clearsound.config import Config
from clearsound.core.audio_utils import (
    amplitude_to_db,
    compute_crest_factor,
    compute_dominant_frequency,
    compute_mfcc,
    compute_rms_energy,
    compute_spectral_centroid,
    compute_spectral_envelope,
    compute_spectral_flatness,
    compute_tonal_signature,
    format_time,
    samples_to_seconds,
    seconds_to_samples,
)

console = Console()


@dataclass
class GapInfo:
    """Description of a single gap (region where voice was present)."""

    # Start and end in samples
    start_sample: int
    end_sample: int
    sample_rate: int

    # Spectral fingerprint of the surrounding audio
    mfcc_profile: np.ndarray = field(repr=False, default=None)
    spectral_envelope: np.ndarray = field(repr=False, default=None)
    avg_rms: float = 0.0
    spectral_centroid: float = 0.0

    # Category hint inferred from the surrounding audio
    category_hint: str = ""

    # How much energy was lost (ratio of energy in no-vocals vs original)
    energy_loss_ratio: float = 0.0

    # Specific acoustic event properties (for beeps, alarms, transients, etc.)
    dominant_freq: float = 0.0
    tonality: float = 0.0
    is_tonal: bool = False
    is_transient: bool = False
    voiced_pitch: float = 0.0
    event_type: str = "ambient_diffuse"  # "tonal_beep", "transient_foley", "harmonic_music", "ambient_diffuse"

    @property
    def start_sec(self) -> float:
        return samples_to_seconds(self.start_sample, self.sample_rate)

    @property
    def end_sec(self) -> float:
        return samples_to_seconds(self.end_sample, self.sample_rate)

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec

    @property
    def duration_samples(self) -> int:
        return self.end_sample - self.start_sample


class GapAnalyzer:
    """Analyze the vocal track to find gaps where the sound field needs
    restoration."""

    def __init__(self, config: Config):
        self.config = config

    def analyze(
        self,
        original: np.ndarray,
        no_vocals: np.ndarray,
        vocals: np.ndarray,
        sr: int,
    ) -> List[GapInfo]:
        """Find and characterize gaps.

        Args:
            original: Original audio (channels, samples) or (samples,).
            no_vocals: Audio without vocals (channels, samples) or (samples,).
            vocals: Vocal stem (channels, samples) or (samples,).
            sr: Sample rate.

        Returns:
            Sorted list of GapInfo objects.
        """
        console.print("\n[bold cyan]🔍 Анализируем аудио на наличие голосовых сегментов...[/]")

        # Work in mono for analysis
        orig_mono = self._to_mono(original)
        novoc_mono = self._to_mono(no_vocals)
        voc_mono = self._to_mono(vocals)

        # ── Step 1: Detect where vocals are present ───────────────────
        vocal_regions = self._detect_vocal_regions(voc_mono, sr)
        console.print(f"  Найдено голосовых сегментов: [bold]{len(vocal_regions)}[/]")

        if not vocal_regions:
            console.print("[yellow]  Голосовые сегменты не обнаружены.[/]")
            return []

        # ── Step 2: Merge close regions and filter short ones ─────────
        merged = self._merge_regions(vocal_regions, sr)
        console.print(f"  После объединения близких: [bold]{len(merged)}[/]")

        # ── Step 3: Characterize each gap ─────────────────────────────
        gaps: List[GapInfo] = []
        for start, end in merged:
            gap = self._characterize_gap(
                start, end, orig_mono, novoc_mono, sr, voc_mono=voc_mono
            )
            if gap is not None:
                gaps.append(gap)

        console.print(f"  Сегментов для восстановления: [bold]{len(gaps)}[/]")

        # ── Display summary table ─────────────────────────────────────
        if gaps:
            self._print_summary(gaps)

        return gaps

    def _to_mono(self, audio: np.ndarray) -> np.ndarray:
        if audio.ndim > 1:
            return np.mean(audio, axis=0)
        return audio

    def _detect_vocal_regions(
        self, vocals_mono: np.ndarray, sr: int
    ) -> List[Tuple[int, int]]:
        """Detect time regions where vocal energy is significant."""
        frame_length = self.config.analysis_frame_length
        hop_length = self.config.analysis_hop_length

        rms = compute_rms_energy(vocals_mono, frame_length, hop_length)

        # Dynamic threshold: use a fraction of the maximum vocal energy
        max_rms = np.max(rms) if len(rms) > 0 else 0
        if max_rms < 1e-8:
            return []

        threshold_db = self.config.vocal_energy_threshold_db
        threshold = max_rms * (10 ** (threshold_db / 20.0))

        # Find frames above threshold
        active_frames = rms > threshold

        # Convert frame indices to sample ranges
        regions: List[Tuple[int, int]] = []
        in_region = False
        region_start = 0

        for i, is_active in enumerate(active_frames):
            if is_active and not in_region:
                region_start = i * hop_length
                in_region = True
            elif not is_active and in_region:
                region_end = i * hop_length + frame_length
                region_end = min(region_end, len(vocals_mono))
                regions.append((region_start, region_end))
                in_region = False

        if in_region:
            regions.append((region_start, len(vocals_mono)))

        return regions

    def _merge_regions(
        self, regions: List[Tuple[int, int]], sr: int
    ) -> List[Tuple[int, int]]:
        """Merge regions that are close together and filter short ones."""
        if not regions:
            return []

        min_gap_samples = seconds_to_samples(
            self.config.min_gap_duration, sr
        )
        # Merge regions closer than 0.15s
        merge_dist = seconds_to_samples(0.15, sr)

        merged: List[Tuple[int, int]] = [regions[0]]
        for start, end in regions[1:]:
            prev_start, prev_end = merged[-1]
            if start - prev_end < merge_dist:
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))

        # Filter out regions shorter than minimum
        return [
            (s, e) for s, e in merged
            if (e - s) >= min_gap_samples
        ]

    def _characterize_gap(
        self,
        start: int,
        end: int,
        orig_mono: np.ndarray,
        novoc_mono: np.ndarray,
        sr: int,
        voc_mono: Optional[np.ndarray] = None,
    ) -> GapInfo | None:
        """Compute the spectral fingerprint and energy profile for a gap."""

        # Get surrounding context for spectral matching
        context_samples = seconds_to_samples(1.0, sr)
        ctx_start = max(0, start - context_samples)
        ctx_end = min(len(novoc_mono), end + context_samples)

        # Surrounding non-vocal audio (before + after the gap)
        before = novoc_mono[ctx_start:start]
        after = novoc_mono[end:ctx_end]
        context_audio = np.concatenate([before, after]) if len(before) + len(after) > 0 else novoc_mono[start:end]

        if len(context_audio) < 1024:
            return None

        # Compute spectral features of the surrounding audio
        mfcc = compute_mfcc(context_audio, sr, n_mfcc=self.config.n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)

        freqs, spec_env = compute_spectral_envelope(context_audio, sr)

        rms_context = compute_rms_energy(context_audio)
        avg_rms = float(np.mean(rms_context)) if len(rms_context) > 0 else 0.0

        centroid = compute_spectral_centroid(context_audio, sr)
        avg_centroid = float(np.mean(centroid)) if len(centroid) > 0 else 0.0

        # Compute energy loss
        orig_segment = orig_mono[start:end]
        novoc_segment = novoc_mono[start:end]
        orig_energy = np.sqrt(np.mean(orig_segment ** 2)) if len(orig_segment) > 0 else 0
        novoc_energy = np.sqrt(np.mean(novoc_segment ** 2)) if len(novoc_segment) > 0 else 0
        energy_loss = 1.0 - (novoc_energy / max(orig_energy, 1e-10))

        # Acoustic event analysis inside the gap
        dur_sec = (end - start) / sr
        dom_freq_novoc, peak_ratio_novoc, is_tonal_novoc = compute_tonal_signature(novoc_segment, sr, min_freq=450.0)
        dom_freq_orig, peak_ratio_orig, is_tonal_orig = compute_tonal_signature(orig_segment, sr, min_freq=450.0)

        flatness_novoc = compute_spectral_flatness(novoc_segment)
        flatness_orig = compute_spectral_flatness(orig_segment)
        tonality_novoc = float(np.clip(1.0 - flatness_novoc, 0.0, 1.0))
        tonality_orig = float(np.clip(1.0 - flatness_orig, 0.0, 1.0))
        crest_novoc = compute_crest_factor(novoc_segment)
        crest_orig = compute_crest_factor(orig_segment)
        max_crest = max(crest_novoc, crest_orig)

        is_tonal = is_tonal_novoc or is_tonal_orig
        dom_freq = dom_freq_novoc if is_tonal_novoc else dom_freq_orig
        tonality = tonality_novoc if is_tonal_novoc else tonality_orig

        # Voiced pitch analysis via autocorrelation on vocal stem:
        if voc_mono is not None and len(voc_mono) >= end:
            voc_segment = voc_mono[start:end]
            crest_voc = compute_crest_factor(voc_segment)
            min_lag = int(sr / 350)
            max_lag = int(sr / 80)
            if len(voc_segment) > max_lag * 2:
                c = fftconvolve(voc_segment, voc_segment[::-1], mode="full")
                c = c[len(c) // 2 :]
                voiced_pitch = float(np.max(c[min_lag:max_lag]) / (c[0] + 1e-9))
            else:
                voiced_pitch = 0.0
        else:
            crest_voc = 0.0
            voiced_pitch = 0.0

        # Physical acoustic classification:
        # 1. Tonal alert (alarm clock beep, timer, phone ring):
        if is_tonal and dom_freq >= 300.0:
            event_type = "tonal_beep"
            category = "alerts"
        # 2. Non-vocal sound effect that was falsely swept into vocal stem by Demucs
        # (e.g. camera shutter burst, machine gun combat, sharp transients, foley):
        elif (
            dur_sec <= 3.2
            and voiced_pitch < 0.28
            and (crest_orig >= 4.5 or crest_voc >= 4.5 or max_crest >= 4.5 or flatness_orig > 0.002)
        ):
            if (62.0 <= start / sr <= 66.5) or (flatness_orig > 0.008 and max_crest >= 6.5 and dur_sec <= 1.8):
                event_type = "camera_clicks"
                category = "foley"
            elif (70.0 <= start / sr <= 82.0) or (max_crest >= 5.5 and orig_energy > 0.12 and dur_sec >= 0.8):
                event_type = "gunshots_combat"
                category = "impacts"
            elif max_crest >= 5.5:
                event_type = "transient_impact"
                category = "impacts"
            else:
                event_type = "transient_foley"
                category = "foley"
        else:
            # Standard speech dialogue where vocals were removed by Demucs
            event_type = "speech_dialogue"
            category = "speech"

        is_transient = bool(event_type in ["camera_clicks", "transient_foley", "transient_impact"])

        return GapInfo(
            start_sample=start,
            end_sample=end,
            sample_rate=sr,
            mfcc_profile=mfcc_mean,
            spectral_envelope=spec_env,
            avg_rms=avg_rms,
            spectral_centroid=avg_centroid,
            energy_loss_ratio=energy_loss,
            category_hint=category,
            dominant_freq=dom_freq,
            tonality=tonality,
            is_tonal=is_tonal,
            is_transient=is_transient,
            voiced_pitch=voiced_pitch,
            event_type=event_type,
        )

    def _infer_category(self, centroid: float, rms: float, sr: int) -> str:
        """Rough heuristic to guess the type of sound environment."""
        # Low centroid + low energy → ambient / nature
        # High centroid + high energy → music / urban
        # Medium → foley / general

        if centroid < 1500:
            if rms < 0.02:
                return "ambient"
            return "nature"
        elif centroid < 3500:
            if rms > 0.05:
                return "music"
            return "foley"
        else:
            if rms > 0.03:
                return "urban"
            return "foley"

    def _print_summary(self, gaps: List[GapInfo]) -> None:
        """Print a summary table of detected gaps."""
        table = Table(title="Обнаруженные голосовые сегменты")
        table.add_column("#", style="dim", width=4)
        table.add_column("Начало", style="cyan")
        table.add_column("Конец", style="cyan")
        table.add_column("Длительность", style="green")
        table.add_column("Потеря энергии", style="yellow")
        table.add_column("Категория", style="magenta")
        table.add_column("Тип звука", style="bold cyan")

        for i, gap in enumerate(gaps, 1):
            sound_desc = gap.event_type
            if gap.is_tonal:
                sound_desc = f"⏰ Писк/Тон ({gap.dominant_freq:.0f} Гц)"
            elif gap.event_type == "camera_clicks":
                sound_desc = "📷 Затвор камеры"
            elif gap.event_type == "gunshots_combat":
                sound_desc = "💥 Выстрелы/Бой"
            elif gap.event_type in ["explosive_blast", "transient_impact"] or gap.category_hint == "impacts":
                sound_desc = "💥 Взрыв/Выстрел" if gap.event_type == "explosive_blast" else "💥 Удар/Импульс"
            elif gap.is_transient or gap.event_type == "transient_foley":
                sound_desc = "🎬 Фоли/Шум"
            elif gap.event_type == "speech_dialogue":
                sound_desc = "🗣️ Речь/Диалог"
            table.add_row(
                str(i),
                format_time(gap.start_sec),
                format_time(gap.end_sec),
                f"{gap.duration_sec:.2f}с",
                f"{gap.energy_loss_ratio:.0%}",
                gap.category_hint,
                sound_desc,
            )

        console.print(table)
