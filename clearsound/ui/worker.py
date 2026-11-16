"""Background Worker Threads for Audio Processing in ClearSound."""

from __future__ import annotations

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
from clearsound.core.gap_analyzer import GapAnalyzer, GapInfo
from clearsound.core.sound_inserter import SoundInserter
from clearsound.core.sound_matcher import MatchResult, SoundMatcher
from clearsound.core.voice_remover import VoiceRemover
from clearsound.library.sound_library import SoundLibrary


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

            # ── Step 2: Gap Analysis ──────────────────────────────────
            self.progress.emit("Анализ голосовых сегментов и спектральных пробелов...", 60)
            analyzer = GapAnalyzer(self.config)
            gaps = analyzer.analyze(original, no_vocals, vocals, sr)

            # ── Step 3: Sound Matching ────────────────────────────────
            self.progress.emit("Поиск подходящих звуков в библиотеке...", 75)
            if self.library.count() == 0:
                self.library.generate_basic_sounds()
            matcher = SoundMatcher(self.config, self.library)
            matches = matcher.find_matches(gaps)

            # ── Step 4: Sound Insertion ───────────────────────────────
            self.progress.emit("Восстановление звучания и вставка звуков...", 85)
            inserter = SoundInserter(self.config)
            enhanced = inserter.insert(no_vocals, gaps, matches, sr)

            enhanced_wav = self.config.temp_dir / f"{self.input_path.stem}_enhanced.wav"
            save_audio(enhanced, enhanced_wav, sr)

            # ── Step 5: Remux Video for Preview (if video input) ──────
            video_exts = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}
            is_video = self.input_path.suffix.lower() in video_exts
            preview_video_path: Optional[Path] = None
            preview_novocal_video_path: Optional[Path] = None

            if is_video:
                self.progress.emit("Подготовка видеопревью с новыми звуками...", 92)
                preview_video_path = self.config.temp_dir / f"{self.input_path.stem}_preview_enhanced.mp4"
                combine_video_audio(self.input_path, enhanced_wav, preview_video_path)

                preview_novocal_video_path = self.config.temp_dir / f"{self.input_path.stem}_preview_novocals.mp4"
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
                "enhanced": enhanced,
                "no_vocals_wav": no_vocals_wav,
                "enhanced_wav": enhanced_wav,
                "preview_video_path": preview_video_path,
                "preview_novocal_video_path": preview_novocal_video_path,
            }
            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))


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
            )

            enhanced_wav = self.config.temp_dir / f"{self.input_path.stem}_enhanced.wav"
            save_audio(enhanced, enhanced_wav, self.sr)

            preview_video_path: Optional[Path] = None
            if self.is_video:
                preview_video_path = self.config.temp_dir / f"{self.input_path.stem}_preview_enhanced.mp4"
                combine_video_audio(self.input_path, enhanced_wav, preview_video_path)

            self.finished.emit({
                "enhanced": enhanced,
                "enhanced_wav": enhanced_wav,
                "preview_video_path": preview_video_path,
            })
        except Exception as e:
            self.error.emit(str(e))
