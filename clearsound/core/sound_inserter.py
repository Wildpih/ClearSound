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
    apply_eq,
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
        sound_overrides: Optional[Dict[int, Path | str]] = None,
        enabled_map: Optional[Dict[int, bool]] = None,
    ) -> np.ndarray:
        """Insert matched sounds into the no-vocals audio.

        Args:
            no_vocals: Audio without vocals, shape (channels, samples) or
                       (samples,).
            gaps: List of gap info from the analyzer.
            matches: Dict gap_index → list of MatchResult (best first).
            sr: Sample rate.
            volume_adjustments: Optional per-gap volume overrides (dB).
            sound_overrides: Optional per-gap sound path overrides.
            enabled_map: Optional mapping gap_index → bool (whether to insert).

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

        for i, gap in enumerate(gaps):
            if enabled_map is not None and not enabled_map.get(i, True):
                continue

            # Determine sound path to use
            sound_path: Optional[Path | str] = None
            if sound_overrides and i in sound_overrides and sound_overrides[i]:
                sound_path = sound_overrides[i]
            else:
                gap_matches = matches.get(i, [])
                if gap_matches:
                    sound_path = gap_matches[0].sound.path

            if not sound_path:
                continue

            # Volume override
            default_vol = 0.0
            gap_matches = matches.get(i, [])
            if gap_matches:
                default_vol = gap_matches[0].recommended_volume_db

            vol_db = (
                volume_adjustments.get(i, default_vol)
                if volume_adjustments
                else default_vol
            )

            try:
                patch = self._prepare_patch_by_path(
                    gap, sound_path, mono_novoc, sr, vol_db
                )
            except Exception as e:
                console.print(f"[dim red]  Пропускаем сегмент {i+1}: {e}[/]")
                continue

            # ── Insert the patch into the result ──────────────────
            start = gap.start_sample
            end = gap.end_sample

            # ── 1. Un-muffle suppression: boost muffled frequency band ───
            if gap.is_tonal and gap.dominant_freq >= 200.0:
                try:
                    boost_db = min(9.0, max(3.0, gap.energy_loss_ratio * 12.0))
                    if is_stereo:
                        for ch in range(result.shape[0]):
                            result[ch, start:end] = apply_eq(result[ch, start:end], sr, gap.dominant_freq, gain_db=boost_db, q=3.5)
                    else:
                        result[start:end] = apply_eq(result[start:end], sr, gap.dominant_freq, gain_db=boost_db, q=3.5)
                except Exception:
                    pass

            # ── 2. Insert the patch into the result ───────────────────────
            patch_len = min(len(patch), end - start)

            if is_stereo:
                for ch in range(result.shape[0]):
                    result[ch, start:start + patch_len] += patch[:patch_len]
            else:
                result[start:start + patch_len] += patch[:patch_len]

            inserted_count += 1

        # ── Normalize to prevent clipping ─────────────────────────────
        max_val = np.max(np.abs(result))
        if max_val > 0.99:
            result = result * (0.95 / max_val)

        console.print(
            f"[bold green]✓ Вставлено звуков: {inserted_count}/{len(gaps)}[/]"
        )
        return result

    def _prepare_patch_by_path(
        self,
        gap: GapInfo,
        sound_path: Path | str,
        mono_novoc: np.ndarray,
        sr: int,
        volume_db: float,
    ) -> np.ndarray:
        """Prepare a sound patch from a path, preserving acoustic nature."""
        audio, _ = load_audio(sound_path, sr=sr, mono=True)
        target_len = gap.duration_samples
        vol_factor = db_to_amplitude(volume_db)

        # ── A. Tonal Sounds (Alarm Beeps, Rings, Chimes) ──────────────
        if gap.is_tonal or gap.event_type == "tonal_beep":
            # Never time-stretch a beep! It distorts pitch and creates robotic warble
            if len(audio) > target_len:
                patch = audio[:target_len] * vol_factor
            else:
                # If short single beep, place it cleanly or repeat with realistic pause
                if len(audio) < int(0.3 * sr) and target_len >= 2 * len(audio):
                    pause = np.zeros(int(0.08 * sr), dtype=np.float32)
                    unit = np.concatenate([audio, pause])
                    repeats = (target_len // len(unit)) + 1
                    patch = np.tile(unit, repeats)[:target_len] * vol_factor
                else:
                    patch = np.zeros(target_len, dtype=np.float32)
                    patch[:len(audio)] = audio * vol_factor

            # Gentle 4ms anti-click fade
            fade_samples = int(0.004 * sr)
            return apply_fade(patch, fade_in_samples=fade_samples, fade_out_samples=fade_samples)

        # ── B. Transient Foley (Knocks, Clicks, Steps) ────────────────
        if gap.is_transient or gap.event_type == "transient_foley":
            # Transients must keep sharp attack
            patch = np.zeros(target_len, dtype=np.float32)
            usable = min(len(audio), target_len)
            patch[:usable] = audio[:usable] * vol_factor
            fade_samples = int(0.003 * sr)
            return apply_fade(patch, fade_in_samples=fade_samples, fade_out_samples=fade_samples)

        # ── C. Continuous Ambient Noise (Room Tone, Wind, Rain) ───────
        if len(audio) < target_len:
            repeats = (target_len // len(audio)) + 1
            audio = np.tile(audio, repeats)

        if len(audio) != target_len:
            ratio = len(audio) / target_len
            if 0.5 < ratio < 2.0:
                audio = time_stretch(audio, target_len, sr)
            else:
                audio = audio[:target_len]

        audio = audio * vol_factor

        # Spectral shaping for ambient background
        if self.config.apply_spectral_shaping and gap.spectral_envelope is not None:
            try:
                audio = spectral_shape(audio, gap.spectral_envelope, sr)
            except Exception:
                pass

        # Match loudness to context
        context_samples = seconds_to_samples(1.0, sr)
        ctx_start = max(0, gap.start_sample - context_samples)
        ctx_end = min(len(mono_novoc), gap.end_sample + context_samples)
        context = np.concatenate([
            mono_novoc[ctx_start:gap.start_sample],
            mono_novoc[gap.end_sample:ctx_end],
        ])

        if len(context) > 0 and np.max(np.abs(context)) > 1e-8:
            audio = match_loudness(audio, context)
            audio = audio * vol_factor

        fade = seconds_to_samples(self.config.crossfade_duration, sr)
        return apply_fade(audio, fade_in_samples=fade, fade_out_samples=fade)

    def _prepare_patch(
        self,
        gap: GapInfo,
        match: MatchResult,
        mono_novoc: np.ndarray,
        sr: int,
        volume_db: float,
    ) -> np.ndarray:
        """Prepare a single sound patch from MatchResult."""
        return self._prepare_patch_by_path(
            gap, match.sound.path, mono_novoc, sr, volume_db
        )
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
