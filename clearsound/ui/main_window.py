"""Main Application Window for ClearSound GUI with Project Hub and Multi-Project Switching."""

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
from clearsound.core.project_manager import Project, ProjectManager
from clearsound.core.sound_matcher import MatchResult
from clearsound.library.sound_library import SoundLibrary
from clearsound.ui.library_widget import LibraryWidget
from clearsound.ui.project_hub_widget import ProjectHubWidget
from clearsound.ui.segment_editor_widget import SegmentEditorWidget
from clearsound.ui.timeline_widget import TimelineWidget
from clearsound.ui.video_player_widget import VideoPlayerWidget
from clearsound.ui.worker import PipelineWorker, QuickReinsertWorker


class MainWindow(QMainWindow):
    """ClearSound Graphical User Interface with Multi-Project Hub."""

    def __init__(self, initial_file: Optional[str] = None):
        super().__init__()
        self.setWindowTitle(f"ClearSound v{__version__} — Удаление голоса и интеллектуальное восстановление звука")
        self.resize(1260, 880)
        self.setAcceptDrops(True)

        self.config = Config()
        self.library = SoundLibrary(self.config)
        self.project_manager = ProjectManager(self.config)

        # Active Project Session
        self.current_project: Optional[Project] = None
        self.original_audio: Optional[np.ndarray] = None
        self.no_vocals_audio: Optional[np.ndarray] = None
        self.vocals_audio: Optional[np.ndarray] = None
        self.enhanced_audio: Optional[np.ndarray] = None
        self.sr: int = 44100

        self.pipeline_worker: Optional[PipelineWorker] = None
        self.reinsert_worker: Optional[QuickReinsertWorker] = None
        self._block_project_combo: bool = False

        self._setup_ui()

        # If an initial file was passed (command line or drag-and-drop to .bat)
        if initial_file:
            p = Path(initial_file)
            if p.exists():
                QTimer.singleShot(100, lambda: self.create_and_open_project(p))
        else:
            # Check existing projects; if any exist, open the hub first
            projects = self.project_manager.list_projects()
            if projects:
                self.tabs.setCurrentIndex(0)  # Open Hub tab

    def _setup_ui(self) -> None:
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 8)
        main_layout.setSpacing(8)

        # ── Top Toolbar ───────────────────────────────────────────────
        toolbar_frame = QFrame()
        toolbar_frame.setStyleSheet(
            "background-color: #141822; border: 1px solid #283244; "
            "border-radius: 8px; padding: 6px;"
        )
        tool_layout = QHBoxLayout(toolbar_frame)
        tool_layout.setContentsMargins(10, 6, 10, 6)
        tool_layout.setSpacing(10)

        # Logo / App title
        lbl_logo = QLabel("🎵 ClearSound")
        lbl_logo.setStyleSheet("font-size: 18px; font-weight: 800; color: #00e5c5; margin-right: 8px;")
        tool_layout.addWidget(lbl_logo)

        # Hub Button
        self.btn_hub = QPushButton("🗂️ Хаб проектов")
        self.btn_hub.setStyleSheet(
            "background-color: #1e2636; color: #00f0cf; font-weight: 700; "
            "border: 1px solid #00d2b4; border-radius: 6px; padding: 7px 14px; font-size: 12px;"
        )
        self.btn_hub.clicked.connect(lambda: self.tabs.setCurrentIndex(0))
        tool_layout.addWidget(self.btn_hub)

        # Active Project Selector
        lbl_proj = QLabel("Проект:")
        lbl_proj.setStyleSheet("color: #94a3b8; font-weight: 600; font-size: 12px; margin-left: 6px;")
        tool_layout.addWidget(lbl_proj)

        self.combo_projects = QComboBox()
        self.combo_projects.setMinimumWidth(220)
        self.combo_projects.currentIndexChanged.connect(self._on_project_combo_changed)
        tool_layout.addWidget(self.combo_projects)

        # New Project / Add file
        self.btn_new_project = QPushButton("➕ Новый ролик")
        self.btn_new_project.setStyleSheet(
            "background-color: #212836; color: #ffffff; font-weight: 700; "
            "border: 1px solid #3d4a61; border-radius: 6px; padding: 7px 14px; font-size: 12px;"
        )
        self.btn_new_project.clicked.connect(self._on_new_file_clicked)
        tool_layout.addWidget(self.btn_new_project)

        tool_layout.addStretch()

        # Action: Process
        self.btn_process = QPushButton("⚡ Запустить обработку")
        self.btn_process.setObjectName("primaryButton")
        self.btn_process.setToolTip("Запустить разделение голоса и интеллектуальное восстановление звука")
        self.btn_process.setStyleSheet(
            "background-color: #00d2b4; color: #071118; font-weight: 800; "
            "border: 1px solid #00f0cf; border-radius: 6px; padding: 7px 20px; font-size: 13px;"
        )
        self.btn_process.clicked.connect(self._on_process_clicked)
        self.btn_process.setEnabled(False)
        tool_layout.addWidget(self.btn_process)

        # Action: Export Audio
        self.btn_export_audio = QPushButton("💾 Сохранить аудио")
        self.btn_export_audio.setStyleSheet(
            "background-color: #212836; color: #ffffff; font-weight: 700; "
            "border: 1px solid #3d4a61; border-radius: 6px; padding: 7px 14px; font-size: 12px;"
        )
        self.btn_export_audio.clicked.connect(self._on_export_audio_clicked)
        self.btn_export_audio.setEnabled(False)
        tool_layout.addWidget(self.btn_export_audio)

        # Action: Export Video
        self.btn_export_video = QPushButton("🎬 Сохранить видео")
        self.btn_export_video.setObjectName("accentButton")
        self.btn_export_video.setStyleSheet(
            "background-color: #8b5cf6; color: #ffffff; font-weight: 800; "
            "border: 1px solid #a78bfa; border-radius: 6px; padding: 7px 18px; font-size: 12px;"
        )
        self.btn_export_video.clicked.connect(self._on_export_video_clicked)
        self.btn_export_video.setEnabled(False)
        tool_layout.addWidget(self.btn_export_video)

        main_layout.addWidget(toolbar_frame)

        # ── Main Tab Widget ───────────────────────────────────────────
        self.tabs = QTabWidget()

        # Tab 0: Project Hub
        self.hub_widget = ProjectHubWidget(self.project_manager)
        self.hub_widget.open_project_requested.connect(self.open_project)
        self.hub_widget.project_deleted.connect(self._on_project_deleted)
        self.hub_widget.project_count_changed.connect(self._on_project_count_changed)
        self.tabs.addTab(self.hub_widget, "🗂️ Хаб проектов")

        # Tab 1: Video Player & Editor
        tab_editor = QWidget()
        editor_layout = QVBoxLayout(tab_editor)
        editor_layout.setContentsMargins(6, 8, 6, 6)
        editor_layout.setSpacing(8)

        editor_splitter = QSplitter(Qt.Vertical)

        # Top area: Video player + Timeline
        top_container = QWidget()
        top_layout = QVBoxLayout(top_container)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(6)

        self.player_widget = VideoPlayerWidget()
        top_layout.addWidget(self.player_widget, stretch=1)

        self.timeline_widget = TimelineWidget()
        top_layout.addWidget(self.timeline_widget)

        editor_splitter.addWidget(top_container)

        # Bottom area: Segment Editor Table
        self.segment_editor = SegmentEditorWidget(self.library)
        editor_splitter.addWidget(self.segment_editor)

        editor_splitter.setSizes([450, 270])
        editor_layout.addWidget(editor_splitter)

        self.tabs.addTab(tab_editor, "🎬 Редактор и Плеер")

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

        self.label_status = QLabel("Готов к работе. Перетащите видеоролик или выберите проект в Хабе.")
        self.label_status.setStyleSheet("color: #94a3b8; font-size: 12px;")
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

        # Sync audio device to library preview player
        self.player_widget.device_changed.connect(
            lambda _: self.library_widget.set_audio_device(self.player_widget.audio_output.device())
        )
        if self.player_widget.audio_output.device():
            self.library_widget.set_audio_device(self.player_widget.audio_output.device())

        self._refresh_projects_combo()

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

    # ── Project Hub & Multi-Project Switching ─────────────────────────

    def _on_project_count_changed(self, count: int) -> None:
        self.btn_hub.setText(f"🗂️ Хаб ({count})")
        self._refresh_projects_combo()

    def _refresh_projects_combo(self) -> None:
        self._block_project_combo = True
        self.combo_projects.clear()

        projects = self.project_manager.list_projects()
        cur_idx = -1

        for i, p in enumerate(projects):
            icon = "🎬 " if p.is_video else "🎵 "
            status_tag = " [✅]" if p.status == "processed" else ""
            self.combo_projects.addItem(f"{icon}{p.name}{status_tag}", p.id)
            if self.current_project and self.current_project.id == p.id:
                cur_idx = i

        if cur_idx >= 0:
            self.combo_projects.setCurrentIndex(cur_idx)
        elif not projects:
            self.combo_projects.addItem("— Нет проектов —", None)

        self._block_project_combo = False

    def _on_project_combo_changed(self, idx: int) -> None:
        if self._block_project_combo:
            return
        proj_id = self.combo_projects.itemData(idx)
        if proj_id:
            self.open_project(proj_id)

    def _on_new_file_clicked(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Добавить ролик или аудиофайл", "",
            "Медиа файлы (*.mp4 *.mkv *.avi *.mov *.webm *.wav *.mp3 *.flac *.m4a);;Все файлы (*.*)"
        )
        if file_path:
            self.create_and_open_project(Path(file_path))

    def create_and_open_project(self, file_path: Path) -> None:
        """Create a new project from a file and open it in the editor."""
        try:
            proj = self.project_manager.create_project(file_path)
            self.hub_widget.refresh_hub()
            self.open_project(proj.id)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка создания проекта", f"Не удалось создать проект:\n{e}")

    def _on_project_deleted(self, project_id: str) -> None:
        """Handle project deletion from hub."""
        self._refresh_projects_combo()
        if self.current_project and self.current_project.id == project_id:
            self.player_widget.pause()
            self.current_project = None
            self.original_audio = None
            self.no_vocals_audio = None
            self.vocals_audio = None
            self.enhanced_audio = None
            self.timeline_widget.set_gaps([])
            self.timeline_widget.set_waveform(None)
            self.segment_editor.load_data([], {})
            self.label_status.setText("Проект удалён. Выберите другой проект в хабе.")
            self.tabs.setCurrentIndex(0)

    def open_project(self, project_id: str) -> None:
        """Switch to and open a project in the video player & editor."""
        # Pause playback if running
        self.player_widget.pause()

        # Save previous project adjustments before switching
        if self.current_project:
            self.project_manager.save_project(self.current_project)

        project = self.project_manager.load_project(project_id)
        if not project:
            QMessageBox.warning(self, "Ошибка", "Не удалось загрузить проект.")
            return

        self.current_project = project

        # Switch to Editor Tab
        self.tabs.setCurrentIndex(1)

        # Update Project Combo Box
        self._block_project_combo = True
        for i in range(self.combo_projects.count()):
            if self.combo_projects.itemData(i) == project.id:
                self.combo_projects.setCurrentIndex(i)
                break
        self._block_project_combo = False

        # Load media in player
        self.player_widget.load_media(
            source_path=project.source_path,
            enhanced_path=project.preview_video_path if project.is_video else project.enhanced_wav,
            no_vocals_path=project.preview_novocal_video_path if project.is_video else project.no_vocals_wav,
        )

        # Update timeline waveform & gaps
        self.timeline_widget.set_waveform(project.waveform_peaks)
        self.timeline_widget.set_gaps(project.gaps, project.enabled_map)

        # Update segment editor
        self.segment_editor.load_data(
            gaps=project.gaps,
            matches=project.matches,
            volume_adjustments=project.volume_adjustments,
            sound_overrides=project.sound_overrides,
            enabled_map=project.enabled_map,
        )

        # Buttons state
        self.btn_process.setEnabled(True)
        is_processed = (project.status == "processed")
        self.btn_export_audio.setEnabled(is_processed)
        self.btn_export_video.setEnabled(is_processed and project.is_video)

        type_str = "Видеоролик" if project.is_video else "Аудиофайл"
        status_str = "Обработан" if is_processed else "Готов к обработке"
        self.label_status.setText(
            f"Открыт проект: «{project.name}» ({type_str}, {project.formatted_duration}) — Статус: {status_str}"
        )

    # ── Drag-and-Drop ─────────────────────────────────────────────────

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        urls = event.mimeData().urls()
        if urls:
            path = Path(urls[0].toLocalFile())
            if path.exists():
                self.create_and_open_project(path)

    # ── Pipeline Processing ───────────────────────────────────────────

    def _on_process_clicked(self) -> None:
        if not self.current_project or not self.current_project.source_path.exists():
            return

        self.btn_process.setEnabled(False)
        self.btn_new_project.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.current_project.status = "processing"
        self.project_manager.save_project(self.current_project)
        self.hub_widget.refresh_hub()

        self.pipeline_worker = PipelineWorker(
            input_path=self.current_project.source_path,
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
        self.btn_new_project.setEnabled(True)
        self.btn_export_audio.setEnabled(True)

        if not self.current_project:
            return

        is_video = self.current_project.is_video
        if is_video:
            self.btn_export_video.setEnabled(True)

        self.original_audio = result["original"]
        self.no_vocals_audio = result["no_vocals"]
        self.vocals_audio = result["vocals"]
        self.enhanced_audio = result["enhanced"]
        self.sr = result["sr"]

        # Update Project Data
        self.current_project.status = "processed"
        self.current_project.gaps = result["gaps"]
        self.current_project.matches = result["matches"]
        self.current_project.waveform_peaks = result["waveform_peaks"]
        self.current_project.no_vocals_wav = result["no_vocals_wav"]
        self.current_project.enhanced_wav = result["enhanced_wav"]
        self.current_project.preview_video_path = result["preview_video_path"]
        self.current_project.preview_novocal_video_path = result["preview_novocal_video_path"]

        # Save to disk
        self.project_manager.save_project(self.current_project)
        self.hub_widget.refresh_hub()
        self._refresh_projects_combo()

        # Update Player track sources
        enhanced_source = result["preview_video_path"] if is_video else result["enhanced_wav"]
        no_vocals_source = result["preview_novocal_video_path"] if is_video else result["no_vocals_wav"]

        self.player_widget.update_track_sources(
            enhanced_path=enhanced_source,
            no_vocals_path=no_vocals_source,
        )

        # Update Timeline
        self.timeline_widget.set_waveform(result["waveform_peaks"])
        self.timeline_widget.set_gaps(self.current_project.gaps)

        # Update Segment Editor
        self.segment_editor.load_data(
            gaps=self.current_project.gaps,
            matches=self.current_project.matches,
        )

        donors_count = len(result.get("donors", []))
        donor_str = f" Найдено {donors_count} чистых звуков-доноров из самого ролика." if donors_count > 0 else ""
        self.label_status.setText(
            f"✅ Проект «{self.current_project.name}» успешно обработан! Обнаружено {len(self.current_project.gaps)} сегментов.{donor_str} "
            f"Проверяйте видео в плеере с новыми звуками."
        )

    def _on_pipeline_error(self, err: str) -> None:
        self.progress_bar.setVisible(False)
        self.btn_process.setEnabled(True)
        self.btn_new_project.setEnabled(True)

        if self.current_project:
            self.current_project.status = "error"
            self.project_manager.save_project(self.current_project)
            self.hub_widget.refresh_hub()

        self.label_status.setText(f"❌ Ошибка обработки: {err}")
        QMessageBox.critical(self, "Ошибка обработки", f"Не удалось обработать проект:\n\n{err}")

    # ── Live Adjustments ──────────────────────────────────────────────

    def _on_segment_changed(self, row: int) -> None:
        if self.current_project:
            self.current_project.volume_adjustments = self.segment_editor.volume_adjustments
            self.current_project.sound_overrides = self.segment_editor.sound_overrides
            self.current_project.enabled_map = self.segment_editor.enabled_map
            self.project_manager.save_project(self.current_project)
            self.timeline_widget.set_gaps(self.current_project.gaps, self.current_project.enabled_map)

    def _on_apply_adjustments(self) -> None:
        """Trigger fast re-render and video remux with user adjustments."""
        if not self.current_project or not self.current_project.no_vocals_wav:
            return

        # Load no_vocals audio if needed
        if self.no_vocals_audio is None and self.current_project.no_vocals_wav.exists():
            from clearsound.core.audio_utils import load_audio
            self.no_vocals_audio, self.sr = load_audio(self.current_project.no_vocals_wav, sr=self.config.sample_rate)

        if self.no_vocals_audio is None:
            return

        self.label_status.setText("Применяем правки звуковой дорожки к проекту...")
        self.segment_editor.btn_apply.setEnabled(False)

        self.reinsert_worker = QuickReinsertWorker(
            config=self.config,
            input_path=self.current_project.source_path,
            no_vocals=self.no_vocals_audio,
            gaps=self.current_project.gaps,
            matches=self.current_project.matches,
            sr=self.sr,
            volume_adjustments=self.segment_editor.volume_adjustments,
            sound_overrides=self.segment_editor.sound_overrides,
            enabled_map=self.segment_editor.enabled_map,
            is_video=self.current_project.is_video,
            vocals=getattr(self, "vocals_audio", None),
        )
        self.reinsert_worker.finished.connect(self._on_reinsert_finished)
        self.reinsert_worker.error.connect(self._on_reinsert_error)
        self.reinsert_worker.start()

    def _on_reinsert_finished(self, res: dict) -> None:
        self.enhanced_audio = res["enhanced"]
        enhanced_source = res["preview_video_path"] if self.current_project.is_video else res["enhanced_wav"]

        if self.current_project:
            self.current_project.enhanced_wav = res["enhanced_wav"]
            self.current_project.preview_video_path = res["preview_video_path"]
            self.project_manager.save_project(self.current_project)

        # Update player track source smoothly without losing timestamp
        self.player_widget.update_track_sources(enhanced_path=enhanced_source)

        self.segment_editor.btn_apply.setEnabled(True)
        self.label_status.setText("✅ Правки применены! Продолжайте просмотр ролика с новыми звуками.")

    def _on_reinsert_error(self, err: str) -> None:
        self.segment_editor.btn_apply.setEnabled(True)
        self.label_status.setText(f"❌ Ошибка обновления: {err}")
        QMessageBox.warning(self, "Ошибка обновления", f"Не удалось обновить дорожку: {err}")

    # ── Export ────────────────────────────────────────────────────────

    def _on_export_audio_clicked(self) -> None:
        if not self.current_project or not self.current_project.enhanced_wav:
            return

        default_name = f"{self.current_project.name}_clearsound.wav"
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить чистую звуковую дорожку", str(self.config.output_dir / default_name),
            "WAV Audio (*.wav);;MP3 Audio (*.mp3)"
        )
        if save_path:
            try:
                import shutil
                shutil.copy2(str(self.current_project.enhanced_wav), save_path)
                QMessageBox.information(self, "Экспорт завершён", f"Аудио успешно сохранено:\n{save_path}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка экспорта", f"Не удалось сохранить аудио: {e}")

    def _on_export_video_clicked(self) -> None:
        if not self.current_project or not self.current_project.is_video or not self.current_project.enhanced_wav:
            return

        default_name = f"{self.current_project.name}_clearsound.mp4"
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить видеоролик с новыми звуками", str(self.config.output_dir / default_name),
            "MP4 Video (*.mp4);;MKV Video (*.mkv)"
        )
        if save_path:
            try:
                combine_video_audio(self.current_project.source_path, self.current_project.enhanced_wav, save_path)
                QMessageBox.information(self, "Экспорт завершён", f"Видеоролик с новыми звуками сохранён:\n{save_path}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка экспорта", f"Не удалось сохранить видео: {e}")
