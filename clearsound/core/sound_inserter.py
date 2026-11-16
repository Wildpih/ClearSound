"""Sound inserter — blends matched library sounds into the vocal-removed
audio, applying crossfades, volume matching, time-stretching, and
optional spectral shaping so the result sounds natural.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn

from clearsound.config import Config
from clearsound.core.audio_utils import (
    apply_fade,
    compute_spectral_envelope,
    crossfade,
    db_to_amplitude,
    load_audio,
    match_loudness,
    save_audio,
    seconds_to_samples,
    spectral_shape,
    time_stretch,
)
from clearsound.core.gap_analyzer import GapInfo
from clearsound.core.sound_matcher import MatchResult

console = Console()


class SoundInserter:
    """Insert matched sounds into the vocal-removed audio."""

    def __init__(self, config: Config):
        self.config = config

    def insert(
        self,
        no_vocals: np.ndarray,
        gaps: List[GapInfo],
        matches: Dict[int, List[MatchResult]],
        sr: int,
        volume_adjustments: Optional[Dict[int, float]] = None,
    ) -> np.ndarray:
        """Insert matched sounds into the no-vocals audio.

        Args:
            no_vocals: Audio without vocals, shape (channels, samples) or
                       (samples,).
            gaps: List of gap info from the analyzer.
            matches: Dict gap_index → list of MatchResult (best first).
            sr: Sample rate.
            volume_adjustments: Optional per-gap volume overrides (dB).

        Returns:
            Enhanced audio with the same shape as no_vocals.
        """
        console.print("\n[bold cyan]🎼 Вставляем звуки в аудио...[/]")

        is_stereo = no_vocals.ndim > 1
        if is_stereo:
            # Work channel by channel
            result = no_vocals.copy()
            mono_novoc = np.mean(no_vocals, axis=0)
        else:
            result = no_vocals.copy()
            mono_novoc = no_vocals.copy()

        fade_samples = seconds_to_samples(self.config.crossfade_duration, sr)

        inserted_count = 0

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("  Вставляем...", total=len(gaps))

            for i, gap in enumerate(gaps):
                gap_matches = matches.get(i, [])
                if not gap_matches:
                    progress.advance(task)
                    continue

                # Use the top match
                best = gap_matches[0]

                # Volume override
                vol_db = (
                    volume_adjustments.get(i, best.recommended_volume_db)
                    if volume_adjustments
                    else best.recommended_volume_db
                )

                try:
                    patch = self._prepare_patch(
                        gap, best, mono_novoc, sr, vol_db
                    )
                except Exception as e:
                    console.print(
                        f"[dim red]  Пропускаем сегмент {i+1}: {e}[/]"
                    )
                    progress.advance(task)
                    continue

                # ── Insert the patch into the result ──────────────────
                start = gap.start_sample
                end = gap.end_sample
                patch_len = min(len(patch), end - start)

                if is_stereo:
                    for ch in range(result.shape[0]):
                        result[ch, start:start + patch_len] += patch[:patch_len]
                else:
                    result[start:start + patch_len] += patch[:patch_len]

                inserted_count += 1
                progress.advance(task)

        # ── Normalize to prevent clipping ─────────────────────────────
        max_val = np.max(np.abs(result))
        if max_val > 0.99:
            result = result * (0.95 / max_val)

        console.print(
            f"[bold green]✓ Вставлено звуков: {inserted_count}/{len(gaps)}[/]"
        )
        return result

    def _prepare_patch(
        self,
        gap: GapInfo,
        match: MatchResult,
        mono_novoc: np.ndarray,
        sr: int,
        volume_db: float,
    ) -> np.ndarray:
        """Prepare a single sound patch for insertion.

        Steps:
        1. Load the matched sound.
        2. Time-stretch to match gap duration.
        3. Apply volume adjustment.
        4. Optionally apply spectral shaping.
        5. Apply fade in/out.
        6. Match loudness to surrounding context.
        """
        # Load
        audio, _ = load_audio(match.sound.path, sr=sr, mono=True)

        target_len = gap.duration_samples

        # ── Time-stretch or loop to match gap length ──────────────────
        if len(audio) < target_len:
            # If the sound is too short, loop it
            repeats = (target_len // len(audio)) + 1
            audio = np.tile(audio, repeats)

        if len(audio) != target_len:
            # Try time stretch for moderate ratios
            ratio = len(audio) / target_len
            if 0.5 < ratio < 2.0:
                audio = time_stretch(audio, target_len, sr)
            else:
                audio = audio[:target_len]

        # ── Volume adjustment ─────────────────────────────────────────
        vol_factor = db_to_amplitude(volume_db)
        audio = audio * vol_factor

        # ── Spectral shaping ──────────────────────────────────────────
        if self.config.apply_spectral_shaping and gap.spectral_envelope is not None:
            try:
                audio = spectral_shape(audio, gap.spectral_envelope, sr)
            except Exception:
                pass  # fall back to unshaped audio

        # ── Match loudness to surrounding context ─────────────────────
        context_samples = seconds_to_samples(1.0, sr)
        ctx_start = max(0, gap.start_sample - context_samples)
        ctx_end = min(len(mono_novoc), gap.end_sample + context_samples)
        context = np.concatenate([
            mono_novoc[ctx_start:gap.start_sample],
            mono_novoc[gap.end_sample:ctx_end],
        ])

        if len(context) > 0 and np.max(np.abs(context)) > 1e-8:
            audio = match_loudness(audio, context)
            # Re-apply the user-requested volume tweak on top
            audio = audio * vol_factor

        # ── Fade in/out ───────────────────────────────────────────────
        fade = seconds_to_samples(self.config.crossfade_duration, sr)
        audio = apply_fade(audio, fade_in_samples=fade, fade_out_samples=fade)

        return audio

    def save_result(
        self,
        audio: np.ndarray,
        output_path: str,
        sr: int,
    ):
        """Save the final enhanced audio."""
        path = save_audio(audio, output_path, sr)
        console.print(f"\n[bold green]💾 Результат сохранён: {path}[/]")
        return path
