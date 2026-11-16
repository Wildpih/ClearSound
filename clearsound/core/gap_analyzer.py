"""Gap analyzer — finds regions where voice was present and the resulting
audio sounds degraded or hollow after vocal removal.

Compares the original audio with the vocal-removed version to identify
segments that need sound restoration, and characterizes the spectral
profile of each gap for later matching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import librosa
import numpy as np
from rich.console import Console
from rich.table import Table

from clearsound.config import Config
from clearsound.core.audio_utils import (
    amplitude_to_db,
    compute_mfcc,
    compute_rms_energy,
    compute_spectral_centroid,
    compute_spectral_envelope,
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

    # Spectral fingerprint of the *surrounding* non-vocal audio
    mfcc_profile: np.ndarray = field(repr=False, default=None)
    spectral_envelope: np.ndarray = field(repr=False, default=None)
    avg_rms: float = 0.0
    spectral_centroid: float = 0.0

    # Category hint inferred from the surrounding audio
    category_hint: str = ""

    # How much energy was lost (ratio of energy in no-vocals vs original)
    energy_loss_ratio: float = 0.0

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
                start, end, orig_mono, novoc_mono, sr
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

        # Infer category from spectral characteristics
        category = self._infer_category(avg_centroid, avg_rms, sr)

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

        for i, gap in enumerate(gaps, 1):
            table.add_row(
                str(i),
                format_time(gap.start_sec),
                format_time(gap.end_sec),
                f"{gap.duration_sec:.2f}с",
                f"{gap.energy_loss_ratio:.0%}",
                gap.category_hint,
            )

        console.print(table)
