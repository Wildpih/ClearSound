"""Common audio utility functions.

Provides loading, saving, spectral analysis, crossfading, EQ, and
other helper functions used throughout the pipeline.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
from scipy import signal as scipy_signal


# ── Loading / Saving ──────────────────────────────────────────────────

def extract_audio_from_video(video_path: str | Path, output_path: Optional[str | Path] = None,
                              sr: int = 44100) -> Path:
    """Extract audio track from a video file using ffmpeg.

    Args:
        video_path: Path to the video file.
        output_path: Where to save the extracted WAV.  Auto-generated if None.
        sr: Target sample rate.

    Returns:
        Path to the extracted WAV file.
    """
    video_path = Path(video_path)
    if output_path is None:
        output_path = video_path.with_suffix(".wav")
    output_path = Path(output_path)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",                    # drop video
        "-acodec", "pcm_s16le",   # 16-bit WAV
        "-ar", str(sr),
        "-ac", "2",               # stereo
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


def combine_video_audio(video_path: str | Path, audio_path: str | Path,
                        output_path: str | Path) -> Path:
    """Remux a video with a new audio track using FFmpeg.

    Copies the video stream without re-encoding (-c:v copy) for maximum
    speed, and encodes the audio as high quality AAC.

    Args:
        video_path: Source video path.
        audio_path: Source audio path (WAV/MP3/etc.).
        output_path: Destination video path (.mp4, .mkv, etc.).

    Returns:
        Path to the remuxed video.
    """
    video_path = Path(video_path)
    audio_path = Path(audio_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "256k",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


def get_waveform_peaks(audio: np.ndarray, num_points: int = 1000) -> np.ndarray:
    """Compute min/max peaks for waveform visualization.

    Args:
        audio: Audio array (mono or multi-channel).
        num_points: Target resolution of peaks.

    Returns:
        Array of shape (num_points, 2) containing [min, max] per segment.
    """
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)

    total_samples = len(audio)
    if total_samples == 0:
        return np.zeros((num_points, 2), dtype=np.float32)

    step = max(1, total_samples // num_points)
    num_bins = min(num_points, total_samples // step)
    peaks = np.zeros((num_bins, 2), dtype=np.float32)

    for i in range(num_bins):
        start = i * step
        end = min(start + step, total_samples)
        segment = audio[start:end]
        if len(segment) > 0:
            peaks[i, 0] = float(np.min(segment))
            peaks[i, 1] = float(np.max(segment))

    return peaks


def load_audio(path: str | Path, sr: int = 44100, mono: bool = False
               ) -> Tuple[np.ndarray, int]:
    """Load an audio file.

    Returns:
        (audio_data, sample_rate) where audio_data has shape (samples,) for
        mono or (channels, samples) for multi-channel.
    """
    path = Path(path)

    # If it's a video format, extract audio first
    video_exts = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}
    if path.suffix.lower() in video_exts:
        wav_path = path.with_suffix(".extracted.wav")
        if not wav_path.exists():
            extract_audio_from_video(path, wav_path, sr)
        path = wav_path

    y, file_sr = sf.read(str(path), always_2d=True)
    # shape: (samples, channels) → transpose to (channels, samples)
    y = y.T

    # Resample if needed
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)

    if mono:
        if y.ndim > 1:
            y = np.mean(y, axis=0)

    return y, sr


def save_audio(audio: np.ndarray, path: str | Path, sr: int = 44100) -> Path:
    """Save audio data to a WAV file.

    Args:
        audio: Audio array of shape (samples,) or (channels, samples).
        path: Output file path.
        sr: Sample rate.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure shape is (samples, channels) for soundfile
    if audio.ndim == 1:
        data = audio
    else:
        data = audio.T  # (channels, samples) → (samples, channels)

    # Clip to prevent distortion
    data = np.clip(data, -1.0, 1.0)
    sf.write(str(path), data, sr)
    return path


# ── Spectral Analysis ─────────────────────────────────────────────────

def compute_mfcc(audio: np.ndarray, sr: int = 44100, n_mfcc: int = 13
                 ) -> np.ndarray:
    """Compute MFCC features for an audio segment.

    Args:
        audio: Mono audio array.
        sr: Sample rate.
        n_mfcc: Number of MFCC coefficients.

    Returns:
        MFCC feature matrix of shape (n_mfcc, n_frames).
    """
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)
    return librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)


def compute_spectral_envelope(audio: np.ndarray, sr: int = 44100,
                               n_fft: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the average spectral envelope of an audio segment.

    Returns:
        (frequencies, magnitudes) arrays.
    """
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)

    S = np.abs(librosa.stft(audio, n_fft=n_fft))
    avg_spectrum = np.mean(S, axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    return freqs, avg_spectrum


def compute_rms_energy(audio: np.ndarray, frame_length: int = 2048,
                       hop_length: int = 512) -> np.ndarray:
    """Compute RMS energy per frame.

    Returns:
        Array of RMS values per frame.
    """
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)
    return librosa.feature.rms(y=audio, frame_length=frame_length,
                               hop_length=hop_length)[0]


def compute_spectral_centroid(audio: np.ndarray, sr: int = 44100) -> np.ndarray:
    """Compute spectral centroid per frame."""
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)
    return librosa.feature.spectral_centroid(y=audio, sr=sr)[0]


def compute_spectral_flatness(audio: np.ndarray, n_fft: int = 2048) -> float:
    """Compute spectral flatness (Wiener entropy).

    Values near 0 indicate strong tonality (beeps, bells, tones, musical notes).
    Values near 1 indicate noise (wind, rain, broadband hiss).
    """
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)
    if len(audio) < 256:
        return 1.0
    n = min(n_fft, len(audio))
    from scipy.fft import rfft
    spec = np.abs(rfft(audio[:n])) ** 2 + 1e-12
    geom_mean = np.exp(np.mean(np.log(spec)))
    arith_mean = np.mean(spec)
    return float(np.clip(geom_mean / (arith_mean + 1e-12), 0.0, 1.0))


def compute_dominant_frequency(audio: np.ndarray, sr: int = 44100, min_freq: float = 200.0) -> Tuple[float, float]:
    """Find dominant spectral peak frequency (Hz) and its prominence ratio over background.

    Returns:
        (dominant_freq_hz, peak_ratio)
    """
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)
    if len(audio) < 256:
        return 0.0, 0.0
    from scipy.fft import rfft, rfftfreq
    spec = np.abs(rfft(audio))
    freqs = rfftfreq(len(audio), 1.0 / sr)
    valid = (freqs >= min_freq) & (freqs <= min(sr / 2.0, 12000.0))
    if not np.any(valid):
        return 0.0, 0.0
    spec_v = spec[valid]
    freqs_v = freqs[valid]
    idx = np.argmax(spec_v)
    peak_val = float(spec_v[idx])
    mean_val = float(np.mean(spec_v)) + 1e-9
    ratio = peak_val / mean_val
    return float(freqs_v[idx]), ratio


def compute_crest_factor(audio: np.ndarray) -> float:
    """Compute peak-to-RMS crest factor to detect impulsive transients (clicks, knocks)."""
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)
    rms = np.sqrt(np.mean(audio ** 2)) + 1e-9
    peak = np.max(np.abs(audio))
    return float(peak / rms)


def amplitude_to_db(amplitude: float) -> float:
    """Convert amplitude to dB."""
    if amplitude <= 0:
        return -np.inf
    return 20.0 * np.log10(amplitude)


def db_to_amplitude(db: float) -> float:
    """Convert dB to amplitude."""
    return 10.0 ** (db / 20.0)


# ── Audio Manipulation ────────────────────────────────────────────────

def crossfade(audio1: np.ndarray, audio2: np.ndarray,
              fade_samples: int) -> np.ndarray:
    """Apply a crossfade between two audio segments.

    The end of audio1 is faded out while the beginning of audio2 is faded in
    over ``fade_samples`` overlapping samples.

    Returns:
        The concatenated result with crossfade applied.
    """
    fade_samples = min(fade_samples, len(audio1), len(audio2))
    if fade_samples <= 0:
        return np.concatenate([audio1, audio2])

    fade_out = np.linspace(1.0, 0.0, fade_samples)
    fade_in = np.linspace(0.0, 1.0, fade_samples)

    result = np.zeros(len(audio1) + len(audio2) - fade_samples)
    result[:len(audio1) - fade_samples] = audio1[:len(audio1) - fade_samples]
    # Overlap region
    overlap_start = len(audio1) - fade_samples
    result[overlap_start:overlap_start + fade_samples] = (
        audio1[-fade_samples:] * fade_out + audio2[:fade_samples] * fade_in
    )
    result[overlap_start + fade_samples:] = audio2[fade_samples:]
    return result


def match_loudness(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Scale *source* so its RMS matches that of *target*."""
    rms_source = np.sqrt(np.mean(source ** 2))
    rms_target = np.sqrt(np.mean(target ** 2))
    if rms_source < 1e-10:
        return source
    return source * (rms_target / rms_source)


def apply_fade(audio: np.ndarray, fade_in_samples: int = 0,
               fade_out_samples: int = 0) -> np.ndarray:
    """Apply fade in / fade out to an audio segment."""
    result = audio.copy()
    if fade_in_samples > 0:
        fade = np.linspace(0.0, 1.0, min(fade_in_samples, len(result)))
        result[:len(fade)] *= fade
    if fade_out_samples > 0:
        fade = np.linspace(1.0, 0.0, min(fade_out_samples, len(result)))
        result[-len(fade):] *= fade
    return result


def time_stretch(audio: np.ndarray, target_length: int,
                 sr: int = 44100) -> np.ndarray:
    """Time-stretch audio to match a target length (in samples)."""
    if len(audio) == 0 or target_length == 0:
        return np.zeros(target_length)
    rate = len(audio) / target_length
    if abs(rate - 1.0) < 0.01:
        # Close enough — just trim or pad
        if len(audio) >= target_length:
            return audio[:target_length]
        return np.pad(audio, (0, target_length - len(audio)))
    stretched = librosa.effects.time_stretch(audio, rate=rate)
    # Trim or pad to exact length
    if len(stretched) >= target_length:
        return stretched[:target_length]
    return np.pad(stretched, (0, target_length - len(stretched)))


def apply_eq(audio: np.ndarray, sr: int, freq: float,
             gain_db: float, q: float = 1.0) -> np.ndarray:
    """Apply a peaking EQ filter at the given frequency.

    Args:
        audio: Mono audio array.
        sr: Sample rate.
        freq: Center frequency in Hz.
        gain_db: Gain in dB (positive = boost, negative = cut).
        q: Quality factor.
    """
    if abs(gain_db) < 0.1:
        return audio

    A = db_to_amplitude(gain_db / 2.0)
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / (2 * q)

    b0 = 1 + alpha * A
    b1 = -2 * np.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha / A

    b = np.array([b0 / a0, b1 / a0, b2 / a0])
    a = np.array([1.0, a1 / a0, a2 / a0])

    return scipy_signal.lfilter(b, a, audio).astype(audio.dtype)


def spectral_shape(source: np.ndarray, target_envelope: np.ndarray,
                   sr: int = 44100, n_fft: int = 2048) -> np.ndarray:
    """Shape the spectrum of *source* to match *target_envelope*.

    This applies a frequency-domain filter so the average spectrum of
    the result resembles *target_envelope*.
    """
    if source.ndim > 1:
        source = np.mean(source, axis=0)

    S = librosa.stft(source, n_fft=n_fft)
    S_mag = np.abs(S)
    S_phase = np.angle(S)

    source_env = np.mean(S_mag, axis=1, keepdims=True)
    source_env = np.maximum(source_env, 1e-10)

    # Ensure target envelope has the right length
    if len(target_envelope) != S_mag.shape[0]:
        target_envelope = np.interp(
            np.linspace(0, 1, S_mag.shape[0]),
            np.linspace(0, 1, len(target_envelope)),
            target_envelope,
        )

    target_env = target_envelope.reshape(-1, 1)
    target_env = np.maximum(target_env, 1e-10)

    # Compute shaping filter (limit to avoid extreme boosts)
    shaping_filter = np.clip(target_env / source_env, 0.1, 10.0)

    S_shaped = S_mag * shaping_filter * np.exp(1j * S_phase)
    return librosa.istft(S_shaped, length=len(source))


def seconds_to_samples(seconds: float, sr: int = 44100) -> int:
    """Convert time in seconds to sample count."""
    return int(seconds * sr)


def samples_to_seconds(samples: int, sr: int = 44100) -> float:
    """Convert sample count to time in seconds."""
    return samples / sr


def format_time(seconds: float) -> str:
    """Format seconds as mm:ss.ms."""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:05.2f}"
