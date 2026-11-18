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
    apply_harmonic_exciter,
    compute_spectral_envelope,
    crossfade,
    db_to_amplitude,
    extract_percussive_transient,
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
        vocals: Optional[np.ndarray] = None,
        music_track: Optional[np.ndarray] = None,
        music_offset_samples: int = 0,
        music_gain: float = 1.0,
        original: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Insert matched sounds into the no-vocals audio.

        Args:
            no_vocals: Audio without vocals, shape (channels, samples) or (samples,).
            gaps: List of gap info from the analyzer.
            matches: Dict gap_index → list of MatchResult (best first).
            sr: Sample rate.
            volume_adjustments: Optional per-gap volume overrides (dB).
            sound_overrides: Optional per-gap sound path overrides.
            enabled_map: Optional mapping gap_index → bool (whether to insert).
            vocals: Optional vocal stem array for HPSS transient de-bleeding.
            music_track: Optional clean background music track (un-muffled, zero voice).
            music_offset_samples: Offset of clean music track relative to start of audio.
            music_gain: Relative gain adjustment for clean music track.
            original: Kept for API compatibility, never directly pasted over gaps.

        Returns:
            Enhanced audio with the same shape as no_vocals.
        """
        console.print("\n[bold cyan]🎼 Вставляем звуки в аудио...[/]")

        is_stereo = no_vocals.ndim > 1
        if is_stereo:
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

            start = gap.start_sample
            end = gap.end_sample
            chunk_novoc = mono_novoc[start:end]
            novoc_rms = float(np.sqrt(np.mean(chunk_novoc ** 2))) if len(chunk_novoc) > 0 else 0.0

            # ── 1. Un-muffle suppression & Restoration for Tonal Alerts ──
            if gap.is_tonal and gap.dominant_freq >= 300.0:
                try:
                    boost_db = min(9.0, max(3.0, gap.energy_loss_ratio * 12.0))
                    if is_stereo:
                        for ch in range(result.shape[0]):
                            result[ch, start:end] = apply_eq(result[ch, start:end], sr, gap.dominant_freq, gain_db=boost_db, q=3.5)
                    else:
                        result[start:end] = apply_eq(result[start:end], sr, gap.dominant_freq, gain_db=boost_db, q=3.5)
                except Exception:
                    pass

            # ── 2. Decide action based on event type ────────────────────────
            sound_path: Optional[Path | str] = None
            is_manual_override = bool(sound_overrides and i in sound_overrides and sound_overrides[i])
            override_val = str(sound_overrides[i]) if is_manual_override else ""

            # Check if user selected clean background music or if gap is speech over aligned music:
            is_speech = (gap.event_type == "speech_dialogue")
            has_aligned_music = (
                music_track is not None
                and start >= music_offset_samples
                and start < (music_offset_samples + (music_track.shape[1] if music_track.ndim > 1 else len(music_track)))
            )

            if override_val == "__clean_music__" or (is_speech and has_aligned_music and not is_manual_override):
                if has_aligned_music and music_track is not None:
                    m_start = start - music_offset_samples
                    m_end = end - music_offset_samples
                    track_len = music_track.shape[1] if music_track.ndim > 1 else len(music_track)
                    m_start_clamped = max(0, min(track_len, m_start))
                    m_end_clamped = max(0, min(track_len, m_end))

                    if m_end_clamped > m_start_clamped:
                        m_dur = m_end_clamped - m_start_clamped
                        if music_track.ndim > 1:
                            m_seg = music_track[:, m_start_clamped:m_end_clamped] * music_gain
                        else:
                            m_seg = music_track[m_start_clamped:m_end_clamped] * music_gain

                        fade_len = min(int(0.04 * sr), m_dur // 4)
                        if is_stereo:
                            if m_seg.ndim == 1:
                                m_seg = np.vstack([m_seg, m_seg])
                            for ch in range(result.shape[0]):
                                seg = m_seg[ch, :m_dur].copy()
                                if fade_len > 0:
                                    w_in = np.linspace(0, 1, fade_len)
                                    seg[:fade_len] = result[ch, start:start+fade_len] * (1 - w_in) + seg[:fade_len] * w_in
                                    seg[-fade_len:] = seg[-fade_len:] * (1 - w_in) + result[ch, start+m_dur-fade_len:start+m_dur] * w_in
                                result[ch, start:start+m_dur] = seg
                        else:
                            seg = (np.mean(m_seg, axis=0) if m_seg.ndim > 1 else m_seg)[:m_dur].copy()
                            if fade_len > 0:
                                w_in = np.linspace(0, 1, fade_len)
                                seg[:fade_len] = result[start:start+fade_len] * (1 - w_in) + seg[:fade_len] * w_in
                                seg[-fade_len:] = seg[-fade_len:] * (1 - w_in) + result[start+m_dur-fade_len:start+m_dur] * w_in
                            result[start:start+m_dur] = seg
                        inserted_count += 1
                        continue

            # Case A: User explicitly selected clean background (no sound)
            if override_val in ["__none__", "none"]:
                continue

            # Case B: Manual sound override (path or sound name from library)
            if is_manual_override and override_val not in ["__restore_orig__", "__clean_music__"]:
                sound_path = sound_overrides[i]
            else:
                # Case C: Automatic matching for SFX gaps
                gap_matches = matches.get(i, [])
                if gap.event_type in ["camera_drop", "camera_clicks", "gunshots_combat", "tonal_beep", "transient_impact", "transient_foley"]:
                    if gap_matches:
                        sound_path = gap_matches[0].sound.path
                elif is_speech:
                    # Normal speech without music: leave background clean
                    continue

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

            # ── 3. Insert the patch into the result ───────────────────────
            patch_len = min(len(patch), end - start)

            if is_stereo:
                for ch in range(result.shape[0]):
                    seg = result[ch, start:start + patch_len] + patch[:patch_len]
                    seg_max = np.max(np.abs(seg))
                    if seg_max > 1.0:
                        seg = seg * (0.98 / seg_max)
                    result[ch, start:start + patch_len] = seg
            else:
                seg = result[start:start + patch_len] + patch[:patch_len]
                seg_max = np.max(np.abs(seg))
                if seg_max > 1.0:
                    seg = seg * (0.98 / seg_max)
                result[start:start + patch_len] = seg

            inserted_count += 1

        # ── Final safety clip without global attenuation ──────────────
        np.clip(result, -1.0, 1.0, out=result)

        console.print(
            f"[bold green]✓ Вставлено звуков: {inserted_count}/{len(gaps)}[/]"
        )
        return result

    def _debleed_vocal(self, audio_seg: np.ndarray, voc_seg: np.ndarray, sr: int = 44100) -> np.ndarray:
        """Attenuate residual vocal formants that Demucs leaked into no_vocals."""
        if len(audio_seg) < 512 or np.max(np.abs(voc_seg)) < 1e-6:
            return audio_seg
        from scipy.signal import stft, istft
        nperseg = min(2048, len(audio_seg))
        noverlap = nperseg * 3 // 4
        _, _, Zn = stft(audio_seg, fs=sr, nperseg=nperseg, noverlap=noverlap)
        _, _, Zv = stft(voc_seg, fs=sr, nperseg=nperseg, noverlap=noverlap)
        mag_n = np.abs(Zn)
        mag_v = np.abs(Zv)
        phase_n = np.angle(Zn)
        freqs = np.linspace(0, sr / 2, mag_n.shape[0])
        vocal_band = (freqs >= 100) & (freqs <= 3500)
        v_peak = np.max(mag_v) + 1e-9
        v_act = np.clip(mag_v / v_peak, 0.0, 1.0)
        gain = np.ones_like(mag_n)
        gain[vocal_band, :] = np.clip(1.0 - 0.40 * v_act[vocal_band, :], 0.35, 1.0)
        Z_clean = (mag_n * gain) * np.exp(1j * phase_n)
        _, out = istft(Z_clean, fs=sr, nperseg=nperseg, noverlap=noverlap)
        return out[:len(audio_seg)].astype(audio_seg.dtype)

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

        # ── B. Camera Drop Impact ─────────────────────────────────────
        if gap.event_type == "camera_drop":
            patch = np.zeros(target_len, dtype=np.float32)
            usable = min(len(audio), target_len)
            patch[:usable] = audio[:usable] * vol_factor
            fade_in = int(0.002 * sr)
            fade_out = int(0.02 * sr)
            return apply_fade(patch, fade_in_samples=fade_in, fade_out_samples=fade_out)

        # ── C. Camera Shutter Clicks (Burst or Single) ────────────────
        if gap.event_type == "camera_clicks":
            if len(audio) < target_len:
                repeats = (target_len // len(audio)) + 1
                patch = np.tile(audio, repeats)[:target_len] * vol_factor
            else:
                patch = audio[:target_len] * vol_factor
            fade = int(0.004 * sr)
            return apply_fade(patch, fade_in_samples=fade, fade_out_samples=fade)

        # ── D. Combat Gunfire & Rifle Bursts ──────────────────────────
        if gap.event_type == "gunshots_combat":
            if len(audio) < target_len:
                repeats = (target_len // len(audio)) + 1
                patch = np.tile(audio, repeats)[:target_len] * vol_factor
            else:
                patch = audio[:target_len] * vol_factor
            fade = int(0.003 * sr)
            return apply_fade(patch, fade_in_samples=fade, fade_out_samples=fade)

        # ── B. Transient Foley (Knocks, Clicks, Steps) ────────────────
        if gap.is_transient or gap.event_type == "transient_foley":
            # Transients must keep sharp attack
            patch = np.zeros(target_len, dtype=np.float32)
            usable = min(len(audio), target_len)
            patch[:usable] = audio[:usable] * vol_factor
            fade_samples = int(0.003 * sr)
            return apply_fade(patch, fade_in_samples=fade_samples, fade_out_samples=fade_samples)

        # ── C. Cinematic Impacts & Explosions (One-Shot with Natural Tail) ───
        if gap.event_type in ["explosive_blast", "transient_impact"] or gap.category_hint == "impacts":
            # Never loop or stretch an explosion/gunshot!
            # Keep authentic attack and let tail decay naturally.
            usable = min(len(audio), target_len)
            patch = np.zeros(target_len, dtype=np.float32)
            patch[:usable] = audio[:usable] * vol_factor
            fade_in = int(0.002 * sr)
            fade_out = int(0.02 * sr)
            return apply_fade(patch, fade_in_samples=fade_in, fade_out_samples=fade_out)

        # ── D. Continuous Ambient Noise (Room Tone, Wind, Rain) ───────
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
