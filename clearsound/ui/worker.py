"""Background Worker Threads for Audio Processing in ClearSound."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal

from clearsound.config import Config
from clearsound.core.audio_utils import (
    combine_video_audio,
    get_waveform_peaks,
    load_audio,
    save_audio,
)
from clearsound.core.donor_extractor import DonorExtractor, DonorSound
from clearsound.core.gap_analyzer import GapAnalyzer, GapInfo
from clearsound.core.sound_inserter import SoundInserter
from clearsound.core.sound_matcher import MatchResult, SoundMatcher
from clearsound.core.voice_remover import VoiceRemover
from clearsound.library.sound_library import SoundLibrary
from rich.console import Console

console = Console()


class PipelineWorker(QThread):
    """Background worker for the full ClearSound processing pipeline."""

    progress = Signal(str, int)     # (status_message, percent 0-100)
    finished = Signal(dict)         # dictionary of results
    error = Signal(str)            # error message string

    def __init__(
        self,
        input_path: Path,
        config: Config,
        library: SoundLibrary,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self.input_path = Path(input_path)
        self.config = config
        self.library = library

    def run(self) -> None:
        try:
            self.progress.emit("Загрузка медиафайла...", 5)

            # ── Step 0: Load original ─────────────────────────────────
            original, sr = load_audio(self.input_path, sr=self.config.sample_rate)
            waveform_peaks = get_waveform_peaks(original, num_points=1200)

            # ── Step 1: Voice Removal ─────────────────────────────────
            self.progress.emit("Удаление голоса (Demucs нейросеть)...", 20)
            remover = VoiceRemover(self.config)

            def status_cb(msg: str):
                self.progress.emit(msg, 35)

            no_vocals, vocals, sr = remover.separate(self.input_path, callback=status_cb)

            # Save initial no_vocals track
            no_vocals_wav = self.config.temp_dir / f"{self.input_path.stem}_no_vocals.wav"
            save_audio(no_vocals, no_vocals_wav, sr)

            # ── Step 2: Gap Analysis & Donor Sound Extraction ─────────
            self.progress.emit("Анализ пауз и поиск чистых звуков-доноров в видео...", 55)
            analyzer = GapAnalyzer(self.config)
            gaps = analyzer.analyze(original, no_vocals, vocals, sr)

            vocal_regions = [(g.start_sample, g.end_sample) for g in gaps]
            donor_extractor = DonorExtractor(self.config.temp_dir)
            donors = donor_extractor.extract_donors(no_vocals, vocal_regions, sr=sr)

            # ── Step 3: Sound Matching ────────────────────────────────
            self.progress.emit("Умный подбор звуков (доноры из ролика + библиотека)...", 75)
            if self.library.count() == 0:
                self.library.generate_basic_sounds()
            matcher = SoundMatcher(self.config, self.library)
            matches = matcher.find_matches(gaps, donors=donors)

            # ── Step 4: Sound Insertion & Clean Music Substitution ─────
            self.progress.emit("Восстановление звучания и вставка звуков...", 85)
            music_track, music_offset, music_gain, music_name = self._find_and_align_music(original, sr)
            if music_track is not None:
                console.print(f"🎵 Найдена чистая фоновая музыка: [bold cyan]{music_name}[/] (смещение {music_offset/sr:.2f}с)")

            inserter = SoundInserter(self.config)
            enhanced = inserter.insert(
                no_vocals,
                gaps,
                matches,
                sr,
                vocals=vocals,
                music_track=music_track,
                music_offset_samples=music_offset,
                music_gain=music_gain,
                original=original,
            )

            enhanced_wav = self.config.temp_dir / f"{self.input_path.stem}_enhanced.wav"
            save_audio(enhanced, enhanced_wav, sr)

            # ── Step 5: Remux Video for Preview (if video input) ──────
            video_exts = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}
            is_video = self.input_path.suffix.lower() in video_exts
            preview_video_path: Optional[Path] = None
            preview_novocal_video_path: Optional[Path] = None

            if is_video:
                self.progress.emit("Подготовка видеопревью с новыми звуками...", 92)
                ts = int(time.time() * 1000)
                preview_video_path = self.config.temp_dir / f"{self.input_path.stem}_preview_enhanced_{ts}.mp4"
                combine_video_audio(self.input_path, enhanced_wav, preview_video_path)

                preview_novocal_video_path = self.config.temp_dir / f"{self.input_path.stem}_preview_novocals_{ts}.mp4"
                combine_video_audio(self.input_path, no_vocals_wav, preview_novocal_video_path)

            self.progress.emit("Готово!", 100)

            result = {
                "input_path": self.input_path,
                "is_video": is_video,
                "sr": sr,
                "original": original,
                "waveform_peaks": waveform_peaks,
                "no_vocals": no_vocals,
                "vocals": vocals,
                "gaps": gaps,
                "matches": matches,
                "donors": donors,
                "enhanced": enhanced,
                "no_vocals_wav": no_vocals_wav,
                "enhanced_wav": enhanced_wav,
                "preview_video_path": preview_video_path,
                "preview_novocal_video_path": preview_novocal_video_path,
                "music_track": music_track,
                "music_offset": music_offset,
                "music_gain": music_gain,
                "music_name": music_name,
            }
            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))

    def _find_and_align_music(
        self, audio: np.ndarray, sr: int
    ) -> Tuple[Optional[np.ndarray], int, float, str]:
        """Detect and align clean instrumental background music from the library."""
        music_dir = self.config.sounds_dir / "music"
        if not music_dir.exists():
            return None, 0, 1.0, ""

        music_files = list(music_dir.glob("*.wav")) + list(music_dir.glob("*.mp3")) + list(music_dir.glob("*.flac"))
        if not music_files:
            return None, 0, 1.0, ""

        import scipy.signal as signal
        audio_mono = np.mean(audio, axis=0) if audio.ndim > 1 else audio
        ds = 4
        audio_ds = audio_mono[::ds]

        best_file = None
        best_peak = 0.0
        best_offset = 0

        for f in music_files:
            try:
                m_raw, sr_m = load_audio(f, sr=sr, mono=True)
                m_ds = m_raw[::ds]
                if len(m_ds) < 100 or len(audio_ds) < 100:
                    continue
                corr = signal.fftconvolve(audio_ds, m_ds[::-1], mode="full")
                peak_idx = int(np.argmax(np.abs(corr)))
                norm_factor = np.linalg.norm(audio_ds) * np.linalg.norm(m_ds) + 1e-9
                peak_val = float(np.abs(corr[peak_idx]) / norm_factor)
                if peak_val > best_peak and peak_val >= 0.04:
                    best_peak = peak_val
                    best_file = f
                    offset_ds = peak_idx - len(m_ds) + 1
                    best_offset = offset_ds * ds
            except Exception:
                continue

        if best_file is None:
            return None, 0, 1.0, ""

        # Refine offset at full sample rate
        try:
            m_full, sr_m = load_audio(best_file, sr=sr, mono=False)
            m_mono = np.mean(m_full, axis=0) if m_full.ndim > 1 else m_full
            win_len = min(int(10.0 * sr), len(m_mono))
            sub_m = m_mono[:win_len]
            search_start = max(0, best_offset - int(0.5 * sr))
            search_end = min(len(audio_mono), best_offset + win_len + int(0.5 * sr))
            sub_a = audio_mono[search_start:search_end]
            if len(sub_a) >= len(sub_m):
                c = signal.fftconvolve(sub_a, sub_m[::-1], mode="valid")
                fine_idx = int(np.argmax(np.abs(c)))
                exact_offset = search_start + fine_idx
            else:
                exact_offset = max(0, best_offset)

            # Compute relative gain by comparing symphonic body of music
            m_start_eval = int(len(m_mono) * 0.4)
            m_end_eval = int(len(m_mono) * 0.8)
            a_start_eval = exact_offset + m_start_eval
            a_end_eval = min(len(audio_mono), exact_offset + m_end_eval)
            eval_len = min(m_end_eval - m_start_eval, a_end_eval - a_start_eval)

            if eval_len > int(sr * 2):
                rms_a = float(np.sqrt(np.mean(audio_mono[a_start_eval : a_start_eval + eval_len] ** 2))) + 1e-6
                rms_m = float(np.sqrt(np.mean(m_mono[m_start_eval : m_start_eval + eval_len] ** 2))) + 1e-6
                gain = float(np.clip((rms_a / rms_m) * 0.82, 0.45, 0.85))
            else:
                gain = 0.80

            return m_full, exact_offset, gain, best_file.stem
        except Exception:
            return None, 0, 1.0, ""


class QuickReinsertWorker(QThread):
    """Fast background worker to re-insert sounds when user makes adjustments."""

    finished = Signal(dict)
    error = Signal(str)

    def __init__(
        self,
        config: Config,
        input_path: Path,
        no_vocals: np.ndarray,
        gaps: List[GapInfo],
        matches: Dict[int, List[MatchResult]],
        sr: int,
        volume_adjustments: Dict[int, float],
        sound_overrides: Dict[int, Path],
        enabled_map: Dict[int, bool],
        is_video: bool,
        vocals: Optional[np.ndarray] = None,
        original: Optional[np.ndarray] = None,
        music_track: Optional[np.ndarray] = None,
        music_offset_samples: int = 0,
        music_gain: float = 1.0,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self.config = config
        self.input_path = input_path
        self.no_vocals = no_vocals
        self.gaps = gaps
        self.matches = matches
        self.sr = sr
        self.volume_adjustments = volume_adjustments
        self.sound_overrides = sound_overrides
        self.enabled_map = enabled_map
        self.is_video = is_video
        self.vocals = vocals
        self.original = original
        self.music_track = music_track
        self.music_offset_samples = music_offset_samples
        self.music_gain = music_gain

    def run(self) -> None:
        try:
            inserter = SoundInserter(self.config)
            enhanced = inserter.insert(
                self.no_vocals,
                self.gaps,
                self.matches,
                self.sr,
                volume_adjustments=self.volume_adjustments,
                sound_overrides=self.sound_overrides,
                enabled_map=self.enabled_map,
                vocals=self.vocals,
                music_track=self.music_track,
                music_offset_samples=self.music_offset_samples,
                music_gain=self.music_gain,
                original=self.original,
            )

            enhanced_wav = self.config.temp_dir / f"{self.input_path.stem}_enhanced.wav"
            save_audio(enhanced, enhanced_wav, self.sr)

            preview_video_path: Optional[Path] = None
            if self.is_video:
                ts = int(time.time() * 1000)
                preview_video_path = self.config.temp_dir / f"{self.input_path.stem}_preview_enhanced_{ts}.mp4"
                combine_video_audio(self.input_path, enhanced_wav, preview_video_path)

            self.finished.emit({
                "enhanced": enhanced,
                "enhanced_wav": enhanced_wav,
                "preview_video_path": preview_video_path,
            })
        except Exception as e:
            self.error.emit(str(e))
