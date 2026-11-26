"""Sound inserter — reconstructs audio by:
1. Preserving original audio where no speech exists.
2. Silencing (fade-to-zero) regions that are speech-only (no background sounds).
3. Blending Demucs no_vocals base + matched library sounds for regions
   where speech overlaps with other sounds.

No hardcoded timestamps or video-specific logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from rich.console import Console

from clearsound.config import Config
from clearsound.core.audio_utils import (
    apply_fade,
    db_to_amplitude,
    load_audio,
    match_loudness,
    save_audio,
    seconds_to_samples,
    time_stretch,
)
from clearsound.core.gap_analyzer import GapInfo
from clearsound.core.sound_matcher import MatchResult

console = Console()


class SoundInserter:
    """Insert matched sounds into the audio, following the new neural-map
    architecture.  Original audio is the base; speech regions are
    reconstructed depending on what the neural net detected underneath."""

    def __init__(self, config: Config):
        self.config = config

    # ──────────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────────

    def insert(
        self,
        no_vocals: np.ndarray,
        gaps: List[GapInfo],
        matches: Dict[int, List[MatchResult]],
        sr: int,
        volume_adjustments: Optional[Dict[int, float]] = None,
        sound_overrides: Optional[Dict[int, Path | str]] = None,
        enabled_map: Optional[Dict[int, bool]] = None,
        vocals: Optional[np.ndarray] = None,
        music_track: Optional[np.ndarray] = None,
        music_offset_samples: int = 0,
        music_gain: float = 1.0,
        original: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Reconstruct audio with matched sounds inserted into speech regions.

        Reconstruction strategy per gap:
        • no overlapping sounds → fade to silence (avoid Demucs artefacts).
        • overlapping sounds exist → use no_vocals as base in that region,
          then layer matched library SFX on top aligned to transients.
        • user override "__none__" → silence.
        • user override "__clean_music__" → fill with clean music track.
        • user override with file path → load and insert that file.
        """
        console.print("\n[bold cyan]🎼 Вставляем звуки в аудио...[/]")

        # Start from original if available (best quality base)
        if original is not None:
            is_stereo = original.ndim > 1
            result = original.copy()
            total_len = original.shape[-1]
        elif no_vocals is not None:
            is_stereo = no_vocals.ndim > 1
            result = no_vocals.copy()
            total_len = no_vocals.shape[-1]
        else:
            raise ValueError("Either original or no_vocals must be provided.")

        fade_len = min(int(0.020 * sr), total_len // 10)
        inserted_count = 0

        for i, gap in enumerate(gaps):
            # ── Skip disabled gaps ────────────────────────────────────
            if enabled_map is not None and not enabled_map.get(i, True):
                continue

            start = gap.start_sample
            end = gap.end_sample
            if start >= total_len or end <= 0:
                continue
            start = max(0, start)
            end = min(total_len, end)
            seg_len = end - start
            if seg_len < 64:
                continue

            # ── Handle user overrides first ───────────────────────────
            is_manual_override = bool(sound_overrides and i in sound_overrides and sound_overrides[i])
            override_val = str(sound_overrides[i]) if is_manual_override else ""

            if override_val in ("__none__", "none"):
                # User explicitly wants silence
                self._fade_to_silence(result, start, end, fade_len, is_stereo)
                continue

            if override_val == "__clean_music__":
                self._fill_with_music(
                    result, start, end, fade_len, is_stereo,
                    music_track, music_offset_samples, music_gain,
                )
                continue

            if is_manual_override and override_val not in ("__none__", "none", "__clean_music__"):
                # Custom sound file
                self._fade_to_silence(result, start, end, fade_len, is_stereo)
                self._insert_sound_from_path(
                    result, gap, Path(override_val), sr, start, end, fade_len,
                    is_stereo, total_len,
                    volume_adjustments.get(i, 0.0) if volume_adjustments else 0.0,
                )
                inserted_count += 1
                continue

            # ── Automatic logic based on neural analysis ──────────────
            sfx_events = [e for e in gap.overlapping_events if e.category not in ("music", "speech")]
            has_music = any(e.category == "music" for e in gap.overlapping_events) or (
                music_track is not None and start >= music_offset_samples
            )

            # Measure actual non-vocal audio energy in this gap
            if no_vocals is not None:
                nv_seg = no_vocals[:, start:end] if no_vocals.ndim > 1 else no_vocals[start:end]
                novoc_rms = float(np.sqrt(np.mean(nv_seg ** 2))) if len(nv_seg) > 0 else 0.0
            else:
                novoc_rms = 0.0

            has_background_sound = novoc_rms >= 0.002 or gap.has_overlapping_sounds

            # Step 1: Background reconstruction
            if has_music and music_track is not None and start >= music_offset_samples:
                # Region has background music and clean music track is available
                self._fill_with_music(
                    result, start, end, fade_len, is_stereo,
                    music_track, music_offset_samples, music_gain,
                )
            elif has_background_sound:
                # Region has real background sounds (typing, foley, room tone, ambience, impacts)
                # NEVER fade to silence! Preserve no_vocals with clean crossfades.
                if no_vocals is not None:
                    self._crossfade_region(result, no_vocals, start, end, fade_len, is_stereo)
                else:
                    self._fade_to_silence(result, start, end, fade_len, is_stereo)
            else:
                # Truly dead silent region (no music and no ambient sounds) -> fade to silence
                self._fade_to_silence(result, start, end, fade_len, is_stereo)

            # Step 2: Layer matched discrete SFX on top (if any)
            gap_matches = matches.get(i, [])
            if not gap_matches:
                continue

            # Pick the best match
            sound_path: Optional[Path] = None
            best_match = gap_matches[0]
            if best_match.sound.path and Path(best_match.sound.path).is_file():
                sound_path = Path(best_match.sound.path)
            else:
                # Try to find the sound in library by name
                p_check = Path(best_match.sound.path)
                candidates = (
                    list(self.config.sounds_dir.glob(f"**/{p_check.stem}.wav"))
                    + list(self.config.sounds_dir.glob(f"**/{p_check.stem}.mp3"))
                )
                if candidates:
                    sound_path = candidates[0]

            if not sound_path:
                continue

            vol_db = (volume_adjustments or {}).get(i, best_match.recommended_volume_db)
            self._insert_sound_from_path(
                result, gap, sound_path, sr, start, end, fade_len,
                is_stereo, total_len, vol_db,
            )
            inserted_count += 1

        # Final safety clip
        np.clip(result, -1.0, 1.0, out=result)

        console.print(f"[bold green]✓ Вставлено звуков: {inserted_count}/{len(gaps)}[/]")
        return result

    # ──────────────────────────────────────────────────────────────────
    #  Private helpers
    # ──────────────────────────────────────────────────────────────────

    def _fade_to_silence(self, result, start, end, fade_len, is_stereo):
        """Smoothly fade the region to zero — avoids Demucs electric murmur."""
        seg_len = end - start
        f_in = min(fade_len, seg_len // 2)
        f_out = min(fade_len, seg_len // 2)
        w_in = np.linspace(1, 0, f_in) if f_in > 0 else np.array([])
        w_out = np.linspace(0, 1, f_out) if f_out > 0 else np.array([])

        if is_stereo:
            for ch in range(result.shape[0]):
                if f_in > 0:
                    result[ch, start:start + f_in] *= w_in
                if end - start > f_in + f_out:
                    result[ch, start + f_in:end - f_out] = 0.0
                if f_out > 0:
                    result[ch, end - f_out:end] *= w_out
        else:
            if f_in > 0:
                result[start:start + f_in] *= w_in
            if end - start > f_in + f_out:
                result[start + f_in:end - f_out] = 0.0
            if f_out > 0:
                result[end - f_out:end] *= w_out

    def _crossfade_region(self, result, no_vocals, start, end, fade_len, is_stereo):
        """Replace a region in result with no_vocals, with crossfades at borders."""
        seg_len = end - start
        f = min(fade_len, seg_len // 2)

        if is_stereo:
            nv = no_vocals if no_vocals.ndim > 1 else np.vstack([no_vocals, no_vocals])
            for ch in range(min(result.shape[0], nv.shape[0])):
                seg = nv[ch, start:end].copy()
                if f > 0:
                    ramp_in = np.linspace(0, 1, f)
                    ramp_out = np.linspace(1, 0, f)
                    seg[:f] = result[ch, start:start + f] * (1 - ramp_in) + seg[:f] * ramp_in
                    seg[-f:] = seg[-f:] * ramp_out + result[ch, end - f:end] * (1 - ramp_out)
                result[ch, start:end] = seg
        else:
            nv_mono = np.mean(no_vocals, axis=0) if no_vocals.ndim > 1 else no_vocals
            seg = nv_mono[start:end].copy()
            if f > 0:
                ramp_in = np.linspace(0, 1, f)
                ramp_out = np.linspace(1, 0, f)
                seg[:f] = result[start:start + f] * (1 - ramp_in) + seg[:f] * ramp_in
                seg[-f:] = seg[-f:] * ramp_out + result[end - f:end] * (1 - ramp_out)
            result[start:end] = seg

    def _fill_with_music(self, result, start, end, fade_len, is_stereo,
                         music_track, music_offset_samples, music_gain):
        """Fill a region with clean background music from library."""
        if music_track is None:
            self._fade_to_silence(result, start, end, fade_len, is_stereo)
            return

        m_s = start - music_offset_samples
        m_e = end - music_offset_samples
        track_len = music_track.shape[-1]

        if m_s < 0 or m_s >= track_len:
            self._fade_to_silence(result, start, end, fade_len, is_stereo)
            return

        m_dur = min(track_len - m_s, end - start)
        if m_dur <= 0:
            self._fade_to_silence(result, start, end, fade_len, is_stereo)
            return

        f = min(fade_len, m_dur // 2)

        if is_stereo:
            if music_track.ndim == 1:
                m_seg = np.vstack([music_track[m_s:m_s + m_dur],
                                   music_track[m_s:m_s + m_dur]]) * music_gain
            else:
                m_seg = music_track[:, m_s:m_s + m_dur] * music_gain
                if m_seg.shape[0] < result.shape[0]:
                    m_seg = np.repeat(m_seg, result.shape[0], axis=0)
            for ch in range(result.shape[0]):
                seg = m_seg[ch, :m_dur].copy()
                if f > 0:
                    ramp_in = np.linspace(0, 1, f)
                    ramp_out = np.linspace(1, 0, f)
                    seg[:f] = result[ch, start:start + f] * (1 - ramp_in) + seg[:f] * ramp_in
                    seg[-f:] = seg[-f:] * ramp_out + result[ch, start + m_dur - f:start + m_dur] * (1 - ramp_out)
                result[ch, start:start + m_dur] = seg
        else:
            if music_track.ndim > 1:
                m_seg = np.mean(music_track[:, m_s:m_s + m_dur], axis=0) * music_gain
            else:
                m_seg = music_track[m_s:m_s + m_dur] * music_gain
            seg = m_seg[:m_dur].copy()
            if f > 0:
                ramp_in = np.linspace(0, 1, f)
                ramp_out = np.linspace(1, 0, f)
                seg[:f] = result[start:start + f] * (1 - ramp_in) + seg[:f] * ramp_in
                seg[-f:] = seg[-f:] * ramp_out + result[start + m_dur - f:start + m_dur] * (1 - ramp_out)
            result[start:start + m_dur] = seg

    def _insert_sound_from_path(self, result, gap, sound_path, sr, start, end,
                                 fade_len, is_stereo, total_len, volume_db):
        """Load a sound file and overlay it at the gap position, aligned
        to the original transient timing."""
        try:
            snd, _ = load_audio(sound_path, sr=sr, mono=False)
        except Exception as e:
            console.print(f"[dim red]  Не удалось загрузить {sound_path}: {e}[/]")
            return

        vol_factor = db_to_amplitude(volume_db)
        mono_result = np.mean(result, axis=0) if result.ndim > 1 else result

        # ── Transient alignment ───────────────────────────────────────
        # Align the sound's peak/onset to the original's peak in this region
        snd_mono = np.mean(snd, axis=0) if snd.ndim > 1 else snd
        orig_slice = mono_result[start:end]

        # Find peak in original region
        p_orig = start + np.argmax(np.abs(orig_slice)) if len(orig_slice) > 0 else start
        # Find peak in replacement sound
        p_snd = np.argmax(np.abs(snd_mono)) if len(snd_mono) > 0 else 0

        # Bound insert_idx so it stays at the gap and doesn't jump way before
        insert_idx = max(start, p_orig - p_snd)
        if insert_idx > end:
            insert_idx = start

        # Determine usable length — prevent one-shots from bleeding indefinitely into later gaps
        snd_len = snd.shape[-1]
        max_duration = max(end - insert_idx + int(1.0 * sr), int(1.2 * sr))
        usable = min(snd_len, max_duration, total_len - insert_idx)
        if usable <= 0:
            return

        # Short fade-in/out on the replacement sound itself
        f_in = min(int(0.003 * sr), usable // 4)
        f_out = min(int(0.010 * sr), usable // 4)

        if is_stereo:
            if snd.ndim == 1:
                snd = np.vstack([snd, snd])
            elif snd.shape[0] < result.shape[0]:
                snd = np.repeat(snd, result.shape[0] // snd.shape[0] + 1, axis=0)
            for ch in range(result.shape[0]):
                patch = snd[ch, :usable].copy() * vol_factor
                if f_in > 0:
                    patch[:f_in] *= np.linspace(0, 1, f_in)
                if f_out > 0:
                    patch[-f_out:] *= np.linspace(1, 0, f_out)
                result[ch, insert_idx:insert_idx + usable] += patch
        else:
            snd_m = np.mean(snd, axis=0) if snd.ndim > 1 else snd
            patch = snd_m[:usable].copy() * vol_factor
            if f_in > 0:
                patch[:f_in] *= np.linspace(0, 1, f_in)
            if f_out > 0:
                patch[-f_out:] *= np.linspace(1, 0, f_out)
            result[insert_idx:insert_idx + usable] += patch

    # ──────────────────────────────────────────────────────────────────
    #  Save helper
    # ──────────────────────────────────────────────────────────────────

    def save_result(self, audio: np.ndarray, output_path: str, sr: int):
        """Save the final enhanced audio."""
        path = save_audio(audio, output_path, sr)
        console.print(f"\n[bold green]💾 Результат сохранён: {path}[/]")
        return path
