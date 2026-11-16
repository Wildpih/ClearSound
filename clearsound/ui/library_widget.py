"""Sound Library Manager Widget for ClearSound GUI."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from clearsound.core.audio_utils import format_time
from clearsound.library.sound_library import SoundEntry, SoundLibrary


class LibraryWidget(QWidget):
    """Widget to browse, preview, add, and generate sounds in the library."""

    library_updated = Signal()

    def __init__(self, library: SoundLibrary, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.library = library

        # Mini player for listening to library sounds
        self.preview_player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.preview_player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(1.0)

        self._setup_ui()
        self.refresh_library()

    def set_audio_device(self, device) -> None:
        """Route library preview to the chosen audio output device."""
        if device:
            self.audio_output.setDevice(device)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Top Bar (Search + Actions)
        top_bar = QHBoxLayout()

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("🔍 Поиск звуков по названию или тегам...")
        self.search_input.textChanged.connect(self._on_search_changed)
        top_bar.addWidget(self.search_input, stretch=1)

        self.btn_add_sound = QPushButton("➕ Добавить звук")
        self.btn_add_sound.clicked.connect(self._on_add_sound_clicked)
        top_bar.addWidget(self.btn_add_sound)

        self.btn_generate = QPushButton("✨ Базовые звуки")
        self.btn_generate.setToolTip("Сгенерировать базовые синтетические звуки если библиотека пуста")
        self.btn_generate.clicked.connect(self._on_generate_clicked)
        top_bar.addWidget(self.btn_generate)

        layout.addLayout(top_bar)

        # Splitter: Categories on left, Sound list on right
        splitter = QSplitter(Qt.Horizontal)

        # Left: Categories Tree
        left_panel = QFrame()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        cat_title = QLabel("Категории")
        cat_title.setStyleSheet("font-weight: bold; color: #94a3b8;")
        left_layout.addWidget(cat_title)

        self.tree_categories = QTreeWidget()
        self.tree_categories.setHeaderHidden(True)
        self.tree_categories.setFixedWidth(160)
        self.tree_categories.itemClicked.connect(self._on_category_clicked)
        left_layout.addWidget(self.tree_categories)
        splitter.addWidget(left_panel)

        # Right: Sounds Table
        right_panel = QFrame()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.table_sounds = QTableWidget()
        self.table_sounds.setColumnCount(5)
        self.table_sounds.setHorizontalHeaderLabels(["Название", "Категория", "Длительность", "Теги", "Действие"])
        hdr = self.table_sounds.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table_sounds.setSelectionBehavior(QTableWidget.SelectRows)
        right_layout.addWidget(self.table_sounds)
        splitter.addWidget(right_panel)

        splitter.setSizes([160, 600])
        layout.addWidget(splitter)

        # Status and Preview Bar
        preview_bar = QFrame()
        preview_bar.setStyleSheet("background-color: #171b24; border-radius: 6px; padding: 4px;")
        p_layout = QHBoxLayout(preview_bar)
        self.label_preview_status = QLabel("Нажмите ▶ чтобы прослушать любой звук")
        self.label_preview_status.setStyleSheet("color: #94a3b8;")
        p_layout.addWidget(self.label_preview_status, stretch=1)

        self.btn_stop_preview = QPushButton("⏹ Стоп")
        self.btn_stop_preview.clicked.connect(self.preview_player.stop)
        p_layout.addWidget(self.btn_stop_preview)

        layout.addWidget(preview_bar)

    def refresh_library(self) -> None:
        """Reload sounds from disk and update tree and table."""
        self.library.scan()
        all_sounds = self.library.list_all()

        # Update categories tree
        self.tree_categories.clear()
        item_all = QTreeWidgetItem(["Все звуки", str(len(all_sounds))])
        item_all.setData(0, Qt.UserRole, "all")
        self.tree_categories.addTopLevelItem(item_all)

        counts = {}
        for s in all_sounds:
            counts[s.category] = counts.get(s.category, 0) + 1

        for cat in sorted(self.library.KNOWN_CATEGORIES):
            c_item = QTreeWidgetItem([cat.capitalize(), str(counts.get(cat, 0))])
            c_item.setData(0, Qt.UserRole, cat)
            self.tree_categories.addTopLevelItem(c_item)

        item_all.setSelected(True)
        self._populate_table(all_sounds)
        self.library_updated.emit()

    def _populate_table(self, sounds: list[SoundEntry]) -> None:
        self.table_sounds.setRowCount(len(sounds))

        for row, sound in enumerate(sounds):
            self.table_sounds.setRowHeight(row, 40)

            # Name
            item_name = QTableWidgetItem(sound.name)
            item_name.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.table_sounds.setItem(row, 0, item_name)

            # Category
            item_cat = QTableWidgetItem(sound.category)
            item_cat.setTextAlignment(Qt.AlignCenter)
            item_cat.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.table_sounds.setItem(row, 1, item_cat)

            # Duration
            dur_str = format_time(sound.duration_sec) if sound.duration_sec > 0 else "—"
            item_dur = QTableWidgetItem(dur_str)
            item_dur.setTextAlignment(Qt.AlignCenter)
            item_dur.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.table_sounds.setItem(row, 2, item_dur)

            # Tags
            item_tags = QTableWidgetItem(", ".join(sound.tags[:4]))
            item_tags.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.table_sounds.setItem(row, 3, item_tags)

            # Play Button
            btn_play = QPushButton("▶ Слушать")
            btn_play.setStyleSheet(
                "background-color: #212836; color: #00f0cf; font-weight: bold; "
                "border: 1px solid #00d2b4; border-radius: 4px; padding: 4px 12px; font-size: 11px;"
            )
            btn_play.setFixedWidth(95)
            btn_play.clicked.connect(lambda _, p=sound.path, n=sound.name: self._play_sound(p, n))
            self.table_sounds.setCellWidget(row, 4, btn_play)

    def _play_sound(self, path: Path, name: str) -> None:
        if path.exists():
            self.preview_player.stop()
            self.preview_player.setSource(QUrl.fromLocalFile(str(path)))
            self.preview_player.play()
            self.label_preview_status.setText(f"Воспроизведение: {name}")

    def _on_category_clicked(self, item: QTreeWidgetItem, col: int) -> None:
        cat = item.data(0, Qt.UserRole)
        if cat == "all":
            sounds = self.library.list_all()
        else:
            sounds = self.library.by_category(cat)
        self._populate_table(sounds)

    def _on_search_changed(self, text: str) -> None:
        if not text.strip():
            self.refresh_library()
        else:
            results = self.library.search(text.strip())
            self._populate_table(results)

    def _on_add_sound_clicked(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Добавить звук в библиотеку", "",
            "Аудио файлы (*.wav *.mp3 *.flac *.ogg *.m4a)"
        )
        if not file_path:
            return

        # Choose category
        from PySide6.QtWidgets import QInputDialog
        cats = sorted(list(self.library.KNOWN_CATEGORIES))
        cat, ok = QInputDialog.getItem(
            self, "Категория", "Выберите категорию звука:", cats, 0, False
        )
        if ok and cat:
            try:
                self.library.add_sound(file_path, category=cat)
                QMessageBox.information(self, "Успешно", f"Звук добавлен в категорию {cat}")
                self.refresh_library()
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось добавить звук: {e}")

    def _on_generate_clicked(self) -> None:
        n = self.library.generate_basic_sounds()
        QMessageBox.information(self, "Генерация звуков", f"Сгенерировано {n} синтетических звуков!")
        self.refresh_library()
