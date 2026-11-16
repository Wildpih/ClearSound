"""Main Application Window for ClearSound GUI."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QIcon
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from clearsound import __version__
from clearsound.config import Config
from clearsound.core.audio_utils import combine_video_audio, format_time, save_audio
from clearsound.core.gap_analyzer import GapInfo
from clearsound.core.sound_matcher import MatchResult
from clearsound.library.sound_library import SoundLibrary
from clearsound.ui.library_widget import LibraryWidget
from clearsound.ui.segment_editor_widget import SegmentEditorWidget
from clearsound.ui.timeline_widget import TimelineWidget
from clearsound.ui.video_player_widget import VideoPlayerWidget
from clearsound.ui.worker import PipelineWorker, QuickReinsertWorker


class MainWindow(QMainWindow):
    """ClearSound Graphical User Interface."""

    def __init__(self, initial_file: Optional[str] = None):
        super().__init__()
        self.setWindowTitle(f"ClearSound v{__version__} — Удаление голоса и восстановление звука")
        self.resize(1200, 850)
        self.setAcceptDrops(True)

        self.config = Config()
        self.library = SoundLibrary(self.config)

        # Session data
        self.current_input_path: Optional[Path] = None
        self.is_video: bool = False
        self.original_audio: Optional[np.ndarray] = None
        self.no_vocals_audio: Optional[np.ndarray] = None
        self.vocals_audio: Optional[np.ndarray] = None
        self.enhanced_audio: Optional[np.ndarray] = None
        self.sr: int = 44100
        self.gaps: List[GapInfo] = []
        self.matches: Dict[int, List[MatchResult]] = {}

        self.pipeline_worker: Optional[PipelineWorker] = None
        self.reinsert_worker: Optional[QuickReinsertWorker] = None

        self._setup_ui()

        # If an initial file was passed (e.g. from command line or drag-and-drop to .bat)
        if initial_file:
            p = Path(initial_file)
            if p.exists():
                QTimer.singleShot(100, lambda: self.load_file(p))

    def _setup_ui(self) -> None:
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 8)
        main_layout.setSpacing(8)

        # ── Top Toolbar ───────────────────────────────────────────────
        toolbar_frame = QFrame()
        toolbar_frame.setStyleSheet("background-color: #171b24; border: 1px solid #232c3c; border-radius: 8px; padding: 4px;")
        tool_layout = QHBoxLayout(toolbar_frame)
        tool_layout.setContentsMargins(8, 4, 8, 4)
        tool_layout.setSpacing(10)

        # Logo / App title
        lbl_logo = QLabel("🎵 ClearSound")
        lbl_logo.setStyleSheet("font-size: 16px; font-weight: bold; color: #00d2b4; margin-right: 12px;")
        tool_layout.addWidget(lbl_logo)

        self.btn_open = QPushButton("📂 Открыть файл...")
        self.btn_open.clicked.connect(self._on_open_file_clicked)
        tool_layout.addWidget(self.btn_open)

        self.btn_process = QPushButton("⚡ Обработать")
        self.btn_process.setObjectName("primaryButton")
        self.btn_process.setToolTip("Запустить разделение голоса и интеллектуальное восстановление")
        self.btn_process.clicked.connect(self._on_process_clicked)
        self.btn_process.setEnabled(False)
        tool_layout.addWidget(self.btn_process)

        tool_layout.addStretch()

        self.btn_export_audio = QPushButton("💾 Экспорт аудио...")
        self.btn_export_audio.clicked.connect(self._on_export_audio_clicked)
        self.btn_export_audio.setEnabled(False)
        tool_layout.addWidget(self.btn_export_audio)

        self.btn_export_video = QPushButton("🎬 Экспорт видео...")
        self.btn_export_video.setObjectName("accentButton")
        self.btn_export_video.clicked.connect(self._on_export_video_clicked)
        self.btn_export_video.setEnabled(False)
        tool_layout.addWidget(self.btn_export_video)

        main_layout.addWidget(toolbar_frame)

        # ── Main Tab Widget ───────────────────────────────────────────
        self.tabs = QTabWidget()

        # Tab 1: Video & Audio Editor
        tab_editor = QWidget()
        editor_layout = QVBoxLayout(tab_editor)
        editor_layout.setContentsMargins(6, 8, 6, 6)
        editor_layout.setSpacing(8)

        # Splitter: Player on top, Segments below
        editor_splitter = QSplitter(Qt.Vertical)

        # Top area: Video player + Timeline
        top_container = QWidget()
        top_layout = QVBoxLayout(top_container)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(6)

        self.player_widget = VideoPlayerWidget()
        top_layout.addWidget(self.player_widget, stretch=1)

        # Timeline Widget
        self.timeline_widget = TimelineWidget()
        top_layout.addWidget(self.timeline_widget)

        editor_splitter.addWidget(top_container)

        # Bottom area: Segment Editor Table
        self.segment_editor = SegmentEditorWidget(self.library)
        editor_splitter.addWidget(self.segment_editor)

        editor_splitter.setSizes([450, 260])
        editor_layout.addWidget(editor_splitter)

        self.tabs.addTab(tab_editor, "🎬 Видео и Плеер")

        # Tab 2: Sound Library
        self.library_widget = LibraryWidget(self.library)
        self.tabs.addTab(self.library_widget, "📚 Библиотека звуков")

        # Tab 3: Settings
        tab_settings = self._create_settings_tab()
        self.tabs.addTab(tab_settings, "⚙️ Настройки")

        main_layout.addWidget(self.tabs, stretch=1)

        # ── Bottom Status & Progress Bar ──────────────────────────────
        status_frame = QFrame()
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(4, 2, 4, 2)

        self.label_status = QLabel("Готов к работе. Перетащите видео или аудио файл сюда.")
        self.label_status.setStyleSheet("color: #94a3b8;")
        status_layout.addWidget(self.label_status, stretch=1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(280)
        self.progress_bar.setVisible(False)
        status_layout.addWidget(self.progress_bar)

        main_layout.addWidget(status_frame)

        # ── Connect Inter-widget Signals ──────────────────────────────
        self.player_widget.position_changed.connect(self.timeline_widget.set_position)
        self.player_widget.duration_changed.connect(self.timeline_widget.set_duration)

        self.timeline_widget.seek_requested.connect(self.player_widget.set_position)
        self.timeline_widget.gap_selected.connect(self.segment_editor.select_row)

        self.segment_editor.seek_requested.connect(self.player_widget.set_position)
        self.segment_editor.apply_requested.connect(self._on_apply_adjustments)
        self.segment_editor.segment_changed.connect(self._on_segment_changed)

    def _create_settings_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(20, 20, 20, 20)

        group = QGroupBox("Параметры нейросети и аудио")
        group.setObjectName("cardPanel")
        form = QFormLayout(group)
        form.setSpacing(14)

        self.combo_model = QComboBox()
        self.combo_model.addItems(["htdemucs (Рекомендуемая)", "htdemucs_ft (Максимальное качество)", "mdx_extra (Быстрая)"])
        form.addRow("Модель Demucs:", self.combo_model)

        self.spin_min_gap = QDoubleSpinBox()
        self.spin_min_gap.setRange(0.1, 5.0)
        self.spin_min_gap.setSingleStep(0.1)
        self.spin_min_gap.setValue(self.config.min_gap_duration)
        self.spin_min_gap.setSuffix(" сек.")
        form.addRow("Мин. длительность пробела:", self.spin_min_gap)

        self.spin_crossfade = QDoubleSpinBox()
        self.spin_crossfade.setRange(0.01, 0.5)
        self.spin_crossfade.setSingleStep(0.01)
        self.spin_crossfade.setValue(self.config.crossfade_duration)
        self.spin_crossfade.setSuffix(" сек.")
        form.addRow("Длительность кроссфейда:", self.spin_crossfade)

        layout.addWidget(group)

        # Save settings button
        btn_save = QPushButton("Сохранить параметры")
        btn_save.setObjectName("primaryButton")
        btn_save.setFixedWidth(200)
        btn_save.clicked.connect(self._on_save_settings)
        layout.addWidget(btn_save)
        layout.addStretch()

        return widget

    def _on_save_settings(self) -> None:
        idx = self.combo_model.currentIndex()
        models = ["htdemucs", "htdemucs_ft", "mdx_extra"]
        self.config.demucs_model = models[idx]
        self.config.min_gap_duration = self.spin_min_gap.value()
        self.config.crossfade_duration = self.spin_crossfade.value()
        QMessageBox.information(self, "Настройки", "Параметры успешно сохранены!")

    # ── File Handling & Drag-and-Drop ─────────────────────────────────

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        urls = event.mimeData().urls()
        if urls:
            path = Path(urls[0].toLocalFile())
            if path.exists():
                self.load_file(path)

    def _on_open_file_clicked(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите видео или аудио файл", "",
            "Медиа файлы (*.mp4 *.mkv *.avi *.mov *.webm *.wav *.mp3 *.flac *.m4a);;Все файлы (*.*)"
        )
        if file_path:
            self.load_file(Path(file_path))

    def load_file(self, path: Path) -> None:
        self.current_input_path = path
        video_exts = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}
        self.is_video = path.suffix.lower() in video_exts

        self.label_status.setText(f"Загружен файл: {path.name} ({'Видео' if self.is_video else 'Аудио'})")
        self.btn_process.setEnabled(True)
        self.btn_export_audio.setEnabled(False)
        self.btn_export_video.setEnabled(False)

        # Load into player in original mode
        self.player_widget.load_media(path)

        # Clear previous session
        self.timeline_widget.set_gaps([])
        self.timeline_widget.set_waveform(None)

    # ── Pipeline Processing ───────────────────────────────────────────

    def _on_process_clicked(self) -> None:
        if not self.current_input_path or not self.current_input_path.exists():
            return

        self.btn_process.setEnabled(False)
        self.btn_open.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.pipeline_worker = PipelineWorker(
            input_path=self.current_input_path,
            config=self.config,
            library=self.library,
        )
        self.pipeline_worker.progress.connect(self._on_pipeline_progress)
        self.pipeline_worker.finished.connect(self._on_pipeline_finished)
        self.pipeline_worker.error.connect(self._on_pipeline_error)
        self.pipeline_worker.start()

    def _on_pipeline_progress(self, msg: str, percent: int) -> None:
        self.label_status.setText(msg)
        self.progress_bar.setValue(percent)

    def _on_pipeline_finished(self, result: dict) -> None:
        self.progress_bar.setVisible(False)
        self.btn_process.setEnabled(True)
        self.btn_open.setEnabled(True)
        self.btn_export_audio.setEnabled(True)
        if self.is_video:
            self.btn_export_video.setEnabled(True)

        self.original_audio = result["original"]
        self.no_vocals_audio = result["no_vocals"]
        self.vocals_audio = result["vocals"]
        self.enhanced_audio = result["enhanced"]
        self.sr = result["sr"]
        self.gaps = result["gaps"]
        self.matches = result["matches"]

        # Update Player track sources
        enhanced_source = result["preview_video_path"] if self.is_video else result["enhanced_wav"]
        no_vocals_source = result["preview_novocal_video_path"] if self.is_video else result["no_vocals_wav"]

        self.player_widget.update_track_sources(
            enhanced_path=enhanced_source,
            no_vocals_path=no_vocals_source,
        )

        # Update Timeline
        self.timeline_widget.set_waveform(result["waveform_peaks"])
        self.timeline_widget.set_gaps(self.gaps)

        # Update Segment Editor
        self.segment_editor.load_data(self.gaps, self.matches)

        self.label_status.setText(
            f"✅ Обработка завершена! Обнаружено {len(self.gaps)} голосовых сегментов. "
            f"Включайте воспроизведение и проверяйте результат с видео."
        )

    def _on_pipeline_error(self, err: str) -> None:
        self.progress_bar.setVisible(False)
        self.btn_process.setEnabled(True)
        self.btn_open.setEnabled(True)
        self.label_status.setText(f"❌ Ошибка обработки: {err}")
        QMessageBox.critical(self, "Ошибка обработки", f"Не удалось обработать файл:\n\n{err}")

    # ── Live Adjustments ──────────────────────────────────────────────

    def _on_segment_changed(self, row: int) -> None:
        # Update timeline enabled status
        self.timeline_widget.set_gaps(self.gaps, self.segment_editor.enabled_map)

    def _on_apply_adjustments(self) -> None:
        """Trigger fast re-render and video remux with user adjustments."""
        if self.no_vocals_audio is None:
            return

        self.label_status.setText("Применяем правки звуковой дорожки...")
        self.segment_editor.btn_apply.setEnabled(False)

        self.reinsert_worker = QuickReinsertWorker(
            config=self.config,
            input_path=self.current_input_path,
            no_vocals=self.no_vocals_audio,
            gaps=self.gaps,
            matches=self.matches,
            sr=self.sr,
            volume_adjustments=self.segment_editor.volume_adjustments,
            sound_overrides=self.segment_editor.sound_overrides,
            enabled_map=self.segment_editor.enabled_map,
            is_video=self.is_video,
        )
        self.reinsert_worker.finished.connect(self._on_reinsert_finished)
        self.reinsert_worker.error.connect(self._on_reinsert_error)
        self.reinsert_worker.start()

    def _on_reinsert_finished(self, res: dict) -> None:
        self.enhanced_audio = res["enhanced"]
        enhanced_source = res["preview_video_path"] if self.is_video else res["enhanced_wav"]

        # Update player track source smoothly without losing timestamp
        self.player_widget.update_track_sources(enhanced_path=enhanced_source)

        self.segment_editor.btn_apply.setEnabled(True)
        self.label_status.setText("✅ Правки применены! Можно продолжать просмотр с новыми звуками.")

    def _on_reinsert_error(self, err: str) -> None:
        self.segment_editor.btn_apply.setEnabled(True)
        self.label_status.setText(f"❌ Ошибка обновления: {err}")
        QMessageBox.warning(self, "Ошибка обновления", f"Не удалось обновить дорожку: {err}")

    # ── Export ────────────────────────────────────────────────────────

    def _on_export_audio_clicked(self) -> None:
        if self.enhanced_audio is None:
            return

        default_name = f"{self.current_input_path.stem}_clearsound.wav"
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить аудиодорожку", str(self.config.output_dir / default_name),
            "WAV Audio (*.wav);;MP3 Audio (*.mp3)"
        )
        if save_path:
            try:
                save_audio(self.enhanced_audio, save_path, self.sr)
                QMessageBox.information(self, "Экспорт завершён", f"Аудио сохранено в:\n{save_path}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка экспорта", f"Не удалось сохранить аудио: {e}")

    def _on_export_video_clicked(self) -> None:
        if self.enhanced_audio is None or not self.is_video:
            return

        default_name = f"{self.current_input_path.stem}_clearsound.mp4"
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить видео с новыми звуками", str(self.config.output_dir / default_name),
            "MP4 Video (*.mp4);;MKV Video (*.mkv)"
        )
        if save_path:
            try:
                temp_wav = self.config.temp_dir / "export_temp.wav"
                save_audio(self.enhanced_audio, temp_wav, self.sr)
                combine_video_audio(self.current_input_path, temp_wav, save_path)
                QMessageBox.information(self, "Экспорт завершён", f"Видео с новыми звуками сохранено в:\n{save_path}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка экспорта", f"Не удалось сохранить видео: {e}")
