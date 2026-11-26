"""Gap analyzer — Integrates neural sound map to identify regions
where voice was present and what other sounds exist alongside it.

No hardcoded timestamps. Uses SoundMapAnalyzer for universal detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from rich.console import Console
from rich.table import Table

from clearsound.config import Config
from clearsound.core.audio_utils import (
    compute_mfcc,
    compute_rms_energy,
    compute_spectral_centroid,
    compute_spectral_envelope,
    format_time,
    samples_to_seconds,
    seconds_to_samples,
)
from clearsound.core.sound_map_analyzer import SoundEvent, SoundMapAnalyzer, classify_label, CATEGORY_ICONS

console = Console()


@dataclass
class GapInfo:
    """Description of a single gap (region where voice was present)."""

    start_sample: int
    end_sample: int
    sample_rate: int

    # Spectral fingerprint of the surrounding audio
    mfcc_profile: np.ndarray = field(repr=False, default=None)
    spectral_envelope: np.ndarray = field(repr=False, default=None)
    avg_rms: float = 0.0
    spectral_centroid: float = 0.0

    # Category hint — now populated by neural network, NOT hardcoded
    category_hint: str = ""

    # Energy loss ratio (original vs no-vocals)
    energy_loss_ratio: float = 0.0

    # Acoustic properties
    dominant_freq: float = 0.0
    tonality: float = 0.0
    is_tonal: bool = False
    is_transient: bool = False
    voiced_pitch: float = 0.0

    # Neural network classification — replaces all hardcoded event types
    event_type: str = "speech_dialogue"
    neural_label: str = ""           # AudioSet label from PANNs
    neural_category: str = ""        # High-level category
    neural_confidence: float = 0.0   # Detection confidence

    # Overlapping non-speech sounds detected by neural net in this region
    overlapping_events: List[SoundEvent] = field(default_factory=list)

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

    @property
    def has_overlapping_sounds(self) -> bool:
        """True if non-speech sounds exist in this speech region."""
        return len(self.overlapping_events) > 0


class GapAnalyzer:
    """Analyze audio to find speech regions and characterize what sounds
    overlap with them, using neural network detection."""

    def __init__(self, config: Config):
        self.config = config
        self._sound_map_analyzer = SoundMapAnalyzer()

    def analyze(
        self,
        original: np.ndarray,
        no_vocals: np.ndarray,
        vocals: np.ndarray,
        sr: int,
        callback=None,
    ) -> Tuple[List[GapInfo], List[SoundEvent], List[Tuple[float, float]]]:
        """Find and characterize gaps using neural sound map.

        Returns:
            (gaps, sound_events, speech_regions)
        """
        orig_mono = self._to_mono(original)
        novoc_mono = self._to_mono(no_vocals)

        # ── Step 1: Neural sound map on ORIGINAL audio + stems ────────
        sound_events, speech_regions = self._sound_map_analyzer.analyze(
            original, sr, vocals=vocals, no_vocals=no_vocals, callback=callback,
        )

        # ── Step 2: Merge close speech regions ────────────────────────
        merged_speech = self._merge_regions(speech_regions, sr)
        console.print(f"  После объединения: [bold]{len(merged_speech)}[/]")

        # ── Step 3: Create GapInfo for each speech region ─────────────
        gaps: List[GapInfo] = []
        for start_sec, end_sec in merged_speech:
            start_sample = max(0, seconds_to_samples(start_sec, sr))
            end_sample = min(len(orig_mono), seconds_to_samples(end_sec, sr))

            if end_sample - start_sample < seconds_to_samples(self.config.min_gap_duration, sr):
                continue

            gap = self._characterize_gap(
                start_sample, end_sample, orig_mono, novoc_mono, sr, sound_events,
            )
            if gap is not None:
                gaps.append(gap)

        console.print(f"  Сегментов для обработки: [bold]{len(gaps)}[/]")
        if gaps:
            self._print_summary(gaps)

        return gaps, sound_events, speech_regions

    def _to_mono(self, audio):
        return np.mean(audio, axis=0) if audio.ndim > 1 else audio

    def _merge_regions(self, regions, sr):
        if not regions:
            return []
        merge_gap = 0.15
        s = sorted(regions, key=lambda r: r[0])
        merged = [s[0]]
        for start, end in s[1:]:
            ps, pe = merged[-1]
            if start - pe < merge_gap:
                merged[-1] = (ps, max(pe, end))
            else:
                merged.append((start, end))
        min_dur = self.config.min_gap_duration
        return [(s, e) for s, e in merged if e - s >= min_dur]

    def _characterize_gap(self, start, end, orig_mono, novoc_mono, sr, sound_events):
        ctx_samples = seconds_to_samples(1.0, sr)
        ctx_start = max(0, start - ctx_samples)
        ctx_end = min(len(novoc_mono), end + ctx_samples)

        before = novoc_mono[ctx_start:start]
        after = novoc_mono[end:ctx_end]
        ctx_audio = np.concatenate([before, after]) if len(before) + len(after) > 0 else novoc_mono[start:end]

        if len(ctx_audio) < 1024:
            return None

        mfcc = compute_mfcc(ctx_audio, sr, n_mfcc=self.config.n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        freqs, spec_env = compute_spectral_envelope(ctx_audio, sr)
        rms = compute_rms_energy(ctx_audio)
        avg_rms = float(np.mean(rms)) if len(rms) > 0 else 0.0
        centroid = compute_spectral_centroid(ctx_audio, sr)
        avg_centroid = float(np.mean(centroid)) if len(centroid) > 0 else 0.0

        orig_seg = orig_mono[start:end]
        novoc_seg = novoc_mono[start:end]
        orig_e = np.sqrt(np.mean(orig_seg ** 2)) if len(orig_seg) > 0 else 0
        novoc_e = np.sqrt(np.mean(novoc_seg ** 2)) if len(novoc_seg) > 0 else 0
        energy_loss = 1.0 - (novoc_e / max(orig_e, 1e-10))

        # Find overlapping non-speech sound events
        gap_start_sec = start / sr
        gap_end_sec = end / sr
        overlapping = [
            ev for ev in sound_events
            if ev.overlaps_with(gap_start_sec, gap_end_sec) and ev.category != "speech"
        ]

        if overlapping:
            # Prioritize discrete SFX (gunshots, impacts, camera clicks, etc.) over continuous music
            sfx = [e for e in overlapping if e.category not in ("music", "speech")]
            if sfx:
                best = max(sfx, key=lambda e: e.confidence)
            else:
                best = max(overlapping, key=lambda e: e.confidence)
            event_type = best.category
            neural_label = best.label
            neural_category = best.category
            neural_confidence = best.confidence
            category_hint = best.category
        else:
            event_type = "speech_dialogue"
            neural_label = ""
            neural_category = "speech"
            neural_confidence = 0.0
            category_hint = "speech"

        return GapInfo(
            start_sample=start, end_sample=end, sample_rate=sr,
            mfcc_profile=mfcc_mean, spectral_envelope=spec_env,
            avg_rms=avg_rms, spectral_centroid=avg_centroid,
            energy_loss_ratio=energy_loss, category_hint=category_hint,
            event_type=event_type, neural_label=neural_label,
            neural_category=neural_category, neural_confidence=neural_confidence,
            overlapping_events=overlapping,
        )

    def _print_summary(self, gaps):
        table = Table(title="Речевые сегменты (нейросетевая классификация)")
        table.add_column("#", style="dim", width=4)
        table.add_column("Начало", style="cyan")
        table.add_column("Конец", style="cyan")
        table.add_column("Длит.", style="green")
        table.add_column("Потеря", style="yellow")
        table.add_column("Звуки на фоне", style="bold cyan")

        for i, gap in enumerate(gaps, 1):
            if gap.has_overlapping_sounds:
                icons = " ".join(set(ev.icon for ev in gap.overlapping_events))
                labels = ", ".join(set(ev.label for ev in gap.overlapping_events[:3]))
                bg_str = f"{icons} {labels}"
            else:
                bg_str = "🔇 Только речь"
            table.add_row(
                str(i), format_time(gap.start_sec), format_time(gap.end_sec),
                f"{gap.duration_sec:.2f}с", f"{gap.energy_loss_ratio:.0%}", bg_str,
            )
        console.print(table)
