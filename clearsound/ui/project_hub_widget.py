"""Project Hub Widget — a central workspace for managing and switching between multiple video projects."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from clearsound.core.project_manager import Project, ProjectManager

STATUS_STYLES = {
    "processed": ("✅ Обработан", "background-color: #0e3020; color: #34d399; border: 1px solid #10b981; font-weight: bold; border-radius: 4px; padding: 2px 8px; font-size: 11px;"),
    "new": ("🆕 Новый (не обработан)", "background-color: #2d2612; color: #fbbf24; border: 1px solid #f59e0b; font-weight: bold; border-radius: 4px; padding: 2px 8px; font-size: 11px;"),
    "processing": ("⏳ Обрабатывается", "background-color: #0c2b36; color: #38bdf8; border: 1px solid #0ea5e9; font-weight: bold; border-radius: 4px; padding: 2px 8px; font-size: 11px;"),
    "error": ("❌ Ошибка", "background-color: #3b1820; color: #f87171; border: 1px solid #ef4444; font-weight: bold; border-radius: 4px; padding: 2px 8px; font-size: 11px;"),
}


class ProjectCard(QFrame):
    """Visual card displaying a single project in the hub."""

    open_requested = Signal(str)
    delete_requested = Signal(str)
    rename_requested = Signal(str)

    def __init__(self, project: Project, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.project = project
        self.setObjectName("cardPanel")
        self.setStyleSheet(
            "QFrame#cardPanel {"
            "  background-color: #151a24; border: 1px solid #283244; "
            "  border-radius: 8px; padding: 12px; margin-bottom: 6px;"
            "}"
            "QFrame#cardPanel:hover {"
            "  border: 1px solid #00d2b4; background-color: #19202d;"
            "}"
        )
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(14)

        # 1. Media Type Icon
        icon_label = QLabel("🎬" if self.project.is_video else "🎵")
        icon_label.setStyleSheet("font-size: 32px; background: transparent;")
        layout.addWidget(icon_label)

        # 2. Main Info (Title, Path, Duration)
        info_layout = QVBoxLayout()
        info_layout.setSpacing(4)

        title_row = QHBoxLayout()
        title_row.setSpacing(8)

        self.lbl_title = QLabel(self.project.name)
        self.lbl_title.setStyleSheet("font-size: 15px; font-weight: 700; color: #ffffff;")
        title_row.addWidget(self.lbl_title)

        # Rename button
        btn_rename = QPushButton("✏️")
        btn_rename.setToolTip("Переименовать проект")
        btn_rename.setFixedSize(26, 26)
        btn_rename.setStyleSheet("background: transparent; border: none; font-size: 13px;")
        btn_rename.clicked.connect(lambda: self.rename_requested.emit(self.project.id))
        title_row.addWidget(btn_rename)

        title_row.addStretch()
        info_layout.addLayout(title_row)

        # Details line: Path & Timestamps
        lbl_path = QLabel(f"Файл: {self.project.source_path.name}")
        lbl_path.setToolTip(str(self.project.source_path))
        lbl_path.setStyleSheet("color: #94a3b8; font-size: 12px;")
        info_layout.addWidget(lbl_path)

        meta_row = QHBoxLayout()
        meta_row.setSpacing(14)

        lbl_dur = QLabel(f"⏱ {self.project.formatted_duration}")
        lbl_dur.setStyleSheet("color: #cbd5e1; font-weight: 600; font-size: 12px;")
        meta_row.addWidget(lbl_dur)

        lbl_gaps = QLabel(f"✂️ Сегментов: {self.project.gaps_count}")
        lbl_gaps.setStyleSheet("color: #00f0cf; font-weight: 600; font-size: 12px;")
        meta_row.addWidget(lbl_gaps)

        lbl_date = QLabel(f"📅 Обновлён: {self.project.updated_at}")
        lbl_date.setStyleSheet("color: #64748b; font-size: 11px;")
        meta_row.addWidget(lbl_date)

        meta_row.addStretch()
        info_layout.addLayout(meta_row)

        layout.addLayout(info_layout, stretch=1)

        # 3. Status Badge
        status_text, status_style = STATUS_STYLES.get(self.project.status, (self.project.status, ""))
        lbl_status = QLabel(status_text)
        lbl_status.setStyleSheet(status_style)
        lbl_status.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl_status)

        # 4. Action Buttons
        btn_open = QPushButton("▶ Открыть проект")
        btn_open.setStyleSheet(
            "background-color: #00d2b4; color: #071118; font-weight: 800; "
            "border: 1px solid #00f0cf; border-radius: 6px; padding: 8px 18px; font-size: 12px;"
        )
        btn_open.clicked.connect(lambda: self.open_requested.emit(self.project.id))
        layout.addWidget(btn_open)

        btn_delete = QPushButton("🗑")
        btn_delete.setToolTip("Удалить проект из хаба")
        btn_delete.setFixedSize(36, 36)
        btn_delete.setStyleSheet(
            "background-color: #2d1820; color: #f87171; border: 1px solid #ef4444; "
            "border-radius: 6px; font-size: 15px;"
        )
        btn_delete.clicked.connect(lambda: self.delete_requested.emit(self.project.id))
        layout.addWidget(btn_delete)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        self.open_requested.emit(self.project.id)


class ProjectHubWidget(QWidget):
    """Central Hub widget displaying all active and saved projects."""

    open_project_requested = Signal(str)  # project_id
    project_deleted = Signal(str)         # project_id
    project_count_changed = Signal(int)

    def __init__(self, project_manager: ProjectManager, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.pm = project_manager
        self.all_projects: List[Project] = []
        self._setup_ui()
        self.refresh_hub()

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        # ── Top Action Header ─────────────────────────────────────────
        top_bar = QFrame()
        top_bar.setStyleSheet(
            "background-color: #141822; border: 1px solid #283244; "
            "border-radius: 8px; padding: 6px;"
        )
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(10, 6, 10, 6)
        top_layout.setSpacing(12)

        title = QLabel("🗂️ Хаб проектов")
        title.setStyleSheet("font-size: 16px; font-weight: 800; color: #00e5c5;")
        top_layout.addWidget(title)

        self.btn_new_project = QPushButton("➕ Новый проект (Добавить ролик)")
        self.btn_new_project.setStyleSheet(
            "background-color: #00d2b4; color: #071118; font-weight: 800; "
            "border: 1px solid #00f0cf; border-radius: 6px; padding: 8px 20px; font-size: 13px;"
        )
        self.btn_new_project.clicked.connect(self._on_new_project_clicked)
        top_layout.addWidget(self.btn_new_project)

        top_layout.addStretch()

        # Search Bar
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("🔍 Поиск по названию проектов...")
        self.search_input.setFixedWidth(240)
        self.search_input.textChanged.connect(self._filter_cards)
        top_layout.addWidget(self.search_input)

        # Filter Combo
        self.combo_filter = QComboBox()
        self.combo_filter.addItems(["Все проекты", "🎬 Только видео", "🎵 Только аудио", "✅ Обработанные", "🆕 Новые"])
        self.combo_filter.currentIndexChanged.connect(self._filter_cards)
        top_layout.addWidget(self.combo_filter)

        self.btn_refresh = QPushButton("🔄 Обновить")
        self.btn_refresh.clicked.connect(self.refresh_hub)
        top_layout.addWidget(self.btn_refresh)

        main_layout.addWidget(top_bar)

        # ── Scrollable Card Area ──────────────────────────────────────
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self.container_widget = QWidget()
        self.container_layout = QVBoxLayout(self.container_widget)
        self.container_layout.setContentsMargins(0, 0, 0, 0)
        self.container_layout.setSpacing(8)
        self.container_layout.addStretch()

        self.scroll_area.setWidget(self.container_widget)
        main_layout.addWidget(self.scroll_area, stretch=1)

    def refresh_hub(self) -> None:
        """Reload all projects from disk and re-render cards."""
        self.all_projects = self.pm.list_projects()
        self.project_count_changed.emit(len(self.all_projects))
        self._filter_cards()

    def _filter_cards(self) -> None:
        # Clear existing cards in layout
        while self.container_layout.count() > 0:
            item = self.container_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        search_text = self.search_input.text().strip().lower()
        filter_mode = self.combo_filter.currentIndex()

        filtered = []
        for p in self.all_projects:
            # Text search
            if search_text and search_text not in p.name.lower() and search_text not in p.source_path.name.lower():
                continue
            # Filter dropdown
            if filter_mode == 1 and not p.is_video:
                continue
            if filter_mode == 2 and p.is_video:
                continue
            if filter_mode == 3 and p.status != "processed":
                continue
            if filter_mode == 4 and p.status != "new":
                continue
            filtered.append(p)

        if not filtered:
            empty_panel = QFrame()
            empty_panel.setObjectName("cardPanel")
            empty_layout = QVBoxLayout(empty_panel)
            empty_layout.setAlignment(Qt.AlignCenter)
            empty_layout.setSpacing(10)

            lbl_icon = QLabel("📂")
            lbl_icon.setStyleSheet("font-size: 48px;")
            lbl_title = QLabel("Нет проектов для отображения")
            lbl_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #cbd5e1;")
            lbl_sub = QLabel("Нажмите «Новый проект» выше, чтобы добавить видеоролик или аудиофайл в хаб")
            lbl_sub.setStyleSheet("color: #64748b; font-size: 13px;")

            empty_layout.addWidget(lbl_icon, alignment=Qt.AlignCenter)
            empty_layout.addWidget(lbl_title, alignment=Qt.AlignCenter)
            empty_layout.addWidget(lbl_sub, alignment=Qt.AlignCenter)

            self.container_layout.addWidget(empty_panel)
        else:
            for proj in filtered:
                card = ProjectCard(proj)
                card.open_requested.connect(self._on_open_project)
                card.delete_requested.connect(self._on_delete_project)
                card.rename_requested.connect(self._on_rename_project)
                self.container_layout.addWidget(card)

        self.container_layout.addStretch()

    def _on_new_project_clicked(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Добавить ролик или аудиофайл в проект", "",
            "Медиа файлы (*.mp4 *.mkv *.avi *.mov *.webm *.wav *.mp3 *.flac *.m4a);;Все файлы (*.*)"
        )
        if not file_path:
            return

        try:
            proj = self.pm.create_project(Path(file_path))
            self.refresh_hub()
            # Open immediately
            self.open_project_requested.emit(proj.id)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка создания проекта", f"Не удалось создать проект:\n{e}")

    def _on_open_project(self, project_id: str) -> None:
        self.open_project_requested.emit(project_id)

    def _on_delete_project(self, project_id: str) -> None:
        proj = self.pm.load_project(project_id)
        name = proj.name if proj else project_id
        reply = QMessageBox.question(
            self, "Удаление проекта",
            f"Вы уверены, что хотите удалить проект «{name}»?\n\nВсе кэшированные стемы и настройки проекта будут удалены.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.pm.delete_project(project_id)
            self.refresh_hub()
            self.project_deleted.emit(project_id)

    def _on_rename_project(self, project_id: str) -> None:
        proj = self.pm.load_project(project_id)
        if not proj:
            return
        new_name, ok = QInputDialog.getText(
            self, "Переименовать проект", "Новое название проекта:", text=proj.name
        )
        if ok and new_name.strip():
            self.pm.rename_project(project_id, new_name.strip())
            self.refresh_hub()
