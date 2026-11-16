"""Segment Editor Widget — manage and tweak restored sounds for each vocal gap."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from clearsound.core.audio_utils import format_time
from clearsound.core.gap_analyzer import GapInfo
from clearsound.core.sound_matcher import MatchResult
from clearsound.library.sound_library import SoundLibrary

CATEGORY_BADGES = {
    "alerts": ("⏰ Alerts", "background-color: #3f2504; color: #facc15; border: 1px solid #eab308; font-weight: bold; border-radius: 4px; padding: 2px 6px;"),
    "donor": ("🎯 Donor", "background-color: #052e16; color: #4ade80; border: 1px solid #22c55e; font-weight: bold; border-radius: 4px; padding: 2px 6px;"),
    "impacts": ("💥 Impacts", "background-color: #450a0a; color: #f87171; border: 1px solid #ef4444; font-weight: bold; border-radius: 4px; padding: 2px 6px;"),
    "ambient": ("🌊 Ambient", "background-color: #0b2d2a; color: #00f0cf; border: 1px solid #00d2b4; font-weight: bold; border-radius: 4px; padding: 2px 6px;"),
    "nature": ("🍃 Nature", "background-color: #0e3020; color: #34d399; border: 1px solid #10b981; font-weight: bold; border-radius: 4px; padding: 2px 6px;"),
    "music": ("🎵 Music", "background-color: #261942; color: #c4b5fd; border: 1px solid #8b5cf6; font-weight: bold; border-radius: 4px; padding: 2px 6px;"),
    "urban": ("🏙️ Urban", "background-color: #3b280c; color: #fcd34d; border: 1px solid #f59e0b; font-weight: bold; border-radius: 4px; padding: 2px 6px;"),
    "foley": ("🔨 Foley", "background-color: #381228; color: #f472b6; border: 1px solid #ec4899; font-weight: bold; border-radius: 4px; padding: 2px 6px;"),
}


class SegmentEditorWidget(QWidget):
    """Table and controls for customizing sound replacement and volume for each gap."""

    segment_changed = Signal(int)              # gap index modified
    seek_requested = Signal(int)               # position in ms
    apply_requested = Signal()                 # user wants to apply changes

    def __init__(self, library: SoundLibrary, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.library = library
        self.gaps: List[GapInfo] = []
        self.matches: Dict[int, List[MatchResult]] = {}
        self.volume_adjustments: Dict[int, float] = {}
        self.sound_overrides: Dict[int, Path] = {}
        self.enabled_map: Dict[int, bool] = {}

        self._block_updates: bool = False
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Header bar
        header_bar = QHBoxLayout()
        title = QLabel("✂️ Голосовые сегменты для восстановления")
        title.setStyleSheet("font-size: 15px; font-weight: 700; color: #00e5c5;")
        header_bar.addWidget(title)

        header_bar.addStretch()

        self.btn_apply = QPushButton("⚡ Применить правки к звуковой дорожке")
        self.btn_apply.setObjectName("primaryButton")
        self.btn_apply.setStyleSheet(
            "background-color: #00d2b4; color: #071118; border: 1px solid #00f0cf; "
            "font-weight: 700; font-size: 13px; padding: 8px 22px; border-radius: 6px;"
        )
        self.btn_apply.clicked.connect(self.apply_requested.emit)
        header_bar.addWidget(self.btn_apply)

        layout.addLayout(header_bar)

        # Segments Table
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "#", "Время", "Категория", "Звук для вставки", "Громкость", "Вкл", "Действие"
        ])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeToContents)

        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.itemSelectionChanged.connect(self._on_table_selection_changed)
        layout.addWidget(self.table)

        # Quick Batch Adjustment Bar
        batch_bar = QFrame()
        batch_bar.setStyleSheet(
            "background-color: #141822; border: 1px solid #283244; "
            "border-radius: 6px; padding: 6px;"
        )
        batch_layout = QHBoxLayout(batch_bar)
        batch_layout.setContentsMargins(10, 6, 10, 6)
        batch_layout.setSpacing(10)

        batch_label = QLabel("Групповые действия:")
        batch_label.setStyleSheet("color: #cbd5e1; font-weight: bold; font-size: 12px;")
        batch_layout.addWidget(batch_label)

        btn_louder_all = QPushButton("🔊 Все +2 dB")
        btn_louder_all.setStyleSheet("background-color: #212836; color: #ffffff; font-weight: 600; padding: 6px 12px;")
        btn_louder_all.clicked.connect(lambda: self._adjust_all_volume(2.0))
        batch_layout.addWidget(btn_louder_all)

        btn_quieter_all = QPushButton("🔉 Все -2 dB")
        btn_quieter_all.setStyleSheet("background-color: #212836; color: #ffffff; font-weight: 600; padding: 6px 12px;")
        btn_quieter_all.clicked.connect(lambda: self._adjust_all_volume(-2.0))
        batch_layout.addWidget(btn_quieter_all)

        btn_enable_all = QPushButton("✅ Включить все")
        btn_enable_all.setStyleSheet("background-color: #16382c; color: #34d399; border: 1px solid #10b981; font-weight: 600; padding: 6px 12px;")
        btn_enable_all.clicked.connect(lambda: self._set_all_enabled(True))
        batch_layout.addWidget(btn_enable_all)

        btn_disable_all = QPushButton("❌ Отключить все")
        btn_disable_all.setStyleSheet("background-color: #3b1820; color: #f87171; border: 1px solid #ef4444; font-weight: 600; padding: 6px 12px;")
        btn_disable_all.clicked.connect(lambda: self._set_all_enabled(False))
        batch_layout.addWidget(btn_disable_all)

        batch_layout.addStretch()
        layout.addWidget(batch_bar)

    def load_data(
        self,
        gaps: List[GapInfo],
        matches: Dict[int, List[MatchResult]],
        volume_adjustments: Optional[Dict[int, float]] = None,
        sound_overrides: Optional[Dict[int, Path]] = None,
        enabled_map: Optional[Dict[int, bool]] = None,
    ) -> None:
        """Populate the table with detected gaps and match suggestions."""
        self._block_updates = True
        self.gaps = gaps or []
        self.matches = matches or {}
        self.volume_adjustments = volume_adjustments or {}
        self.sound_overrides = sound_overrides or {}
        self.enabled_map = enabled_map or {i: True for i in range(len(self.gaps))}

        self.table.setRowCount(len(self.gaps))

        for i, gap in enumerate(self.gaps):
            # Generous row height so all controls fit comfortably without clipping
            self.table.setRowHeight(i, 44)

            # Column 0: Index
            item_idx = QTableWidgetItem(str(i + 1))
            item_idx.setTextAlignment(Qt.AlignCenter)
            item_idx.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.table.setItem(i, 0, item_idx)

            # Column 1: Time
            time_str = f"{format_time(gap.start_sec)} - {format_time(gap.end_sec)} ({gap.duration_sec:.1f}с)"
            item_time = QTableWidgetItem(time_str)
            item_time.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.table.setItem(i, 1, item_time)

            # Column 2: Category badge
            cat_name = gap.category_hint.lower()
            badge_text, badge_style = CATEGORY_BADGES.get(cat_name, (gap.category_hint, "color: #ffffff;"))
            lbl_cat = QLabel(badge_text)
            lbl_cat.setStyleSheet(badge_style)
            lbl_cat.setAlignment(Qt.AlignCenter)
            cat_container = QWidget()
            cat_layout = QHBoxLayout(cat_container)
            cat_layout.setContentsMargins(6, 4, 6, 4)
            cat_layout.addWidget(lbl_cat)
            self.table.setCellWidget(i, 2, cat_container)

            # Column 3: Sound Selector Combo
            combo_sound = QComboBox()
            combo_sound.setStyleSheet(
                "background-color: #1a2230; color: #ffffff; font-weight: 600; "
                "border: 1px solid #3d4d68; border-radius: 6px; padding: 4px 8px;"
            )
            gap_matches = self.matches.get(i, [])

            # Add match recommendations
            for match in gap_matches:
                score_pct = int(match.similarity_score * 100)
                if match.is_donor:
                    prefix = "" if match.sound.name.startswith("🎯") else "🎯 "
                    label = f"{prefix}{match.sound.name} [{score_pct}% совпадение]"
                else:
                    label = f"⭐ {match.sound.name} ({match.sound.category}) [{score_pct}%]"
                combo_sound.addItem(label, str(match.sound.path))

            # Add library separator and all library sounds
            combo_sound.insertSeparator(combo_sound.count())
            all_lib = self.library.list_all()
            for sound in all_lib:
                if not any(m.sound.path == sound.path for m in gap_matches):
                    combo_sound.addItem(f"📁 {sound.name} ({sound.category})", str(sound.path))

            # Add custom browse option
            combo_sound.insertSeparator(combo_sound.count())
            combo_sound.addItem("➕ Выбрать другой аудиофайл...", "__browse__")

            # Set current selection if override exists
            if i in self.sound_overrides:
                cur_path = str(self.sound_overrides[i])
                idx = combo_sound.findData(cur_path)
                if idx >= 0:
                    combo_sound.setCurrentIndex(idx)
                else:
                    combo_sound.insertItem(0, f"📄 {Path(cur_path).name}", cur_path)
                    combo_sound.setCurrentIndex(0)

            combo_sound.currentIndexChanged.connect(
                lambda idx, row=i, cb=combo_sound: self._on_sound_combo_changed(row, cb)
            )
            self.table.setCellWidget(i, 3, combo_sound)

            # Column 4: Volume Control (Slider + SpinBox)
            vol_widget = QWidget()
            vol_layout = QHBoxLayout(vol_widget)
            vol_layout.setContentsMargins(6, 2, 6, 2)
            vol_layout.setSpacing(6)

            vol_slider = QSlider(Qt.Horizontal)
            vol_slider.setRange(-24, 12)
            vol_slider.setFixedWidth(80)

            vol_spin = QDoubleSpinBox()
            vol_spin.setRange(-24.0, 12.0)
            vol_spin.setSingleStep(0.5)
            vol_spin.setSuffix(" dB")
            vol_spin.setFixedWidth(78)
            vol_spin.setStyleSheet("color: #ffffff; font-weight: bold; background-color: #1a2230;")

            default_vol = 0.0
            if gap_matches:
                default_vol = gap_matches[0].recommended_volume_db
            current_vol = self.volume_adjustments.get(i, default_vol)
            self.volume_adjustments[i] = current_vol

            vol_slider.setValue(int(current_vol))
            vol_spin.setValue(current_vol)

            vol_slider.valueChanged.connect(
                lambda val, spin=vol_spin, row=i: self._on_vol_slider_changed(row, val, spin)
            )
            vol_spin.valueChanged.connect(
                lambda val, slider=vol_slider, row=i: self._on_vol_spin_changed(row, val, slider)
            )

            vol_layout.addWidget(vol_slider)
            vol_layout.addWidget(vol_spin)
            self.table.setCellWidget(i, 4, vol_widget)

            # Column 5: Enabled Checkbox
            check_enabled = QCheckBox()
            check_enabled.setChecked(self.enabled_map.get(i, True))
            check_enabled.toggled.connect(
                lambda state, row=i: self._on_enabled_toggled(row, state)
            )
            check_widget = QWidget()
            check_layout = QHBoxLayout(check_widget)
            check_layout.setContentsMargins(0, 0, 0, 0)
            check_layout.setAlignment(Qt.AlignCenter)
            check_layout.addWidget(check_enabled)
            self.table.setCellWidget(i, 5, check_widget)

            # Column 6: Action Buttons
            act_widget = QWidget()
            act_layout = QHBoxLayout(act_widget)
            act_layout.setContentsMargins(6, 2, 6, 2)
            act_layout.setSpacing(6)

            btn_jump = QPushButton("🎯 Перейти")
            btn_jump.setToolTip("Переместить плеер к началу этого сегмента")
            btn_jump.setStyleSheet(
                "background-color: #212836; color: #ffffff; font-weight: 700; "
                "border: 1px solid #414f68; border-radius: 4px; padding: 5px 12px; font-size: 11px;"
            )
            btn_jump.clicked.connect(lambda _, row=i: self._on_jump_clicked(row))
            act_layout.addWidget(btn_jump)

            self.table.setCellWidget(i, 6, act_widget)

        self._block_updates = False

    def select_row(self, row: int) -> None:
        if 0 <= row < self.table.rowCount():
            self.table.selectRow(row)

    def _on_table_selection_changed(self) -> None:
        selected = self.table.selectedItems()
        if selected:
            row = selected[0].row()
            if 0 <= row < len(self.gaps):
                gap = self.gaps[row]
                pos_ms = int(gap.start_sec * 1000)
                self.seek_requested.emit(pos_ms)

    def _on_sound_combo_changed(self, row: int, combo: QComboBox) -> None:
        if self._block_updates:
            return
        data = combo.currentData()
        if data == "__browse__":
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Выберите аудиофайл для замены", "",
                "Аудио файлы (*.wav *.mp3 *.flac *.ogg *.m4a)"
            )
            if file_path:
                p = Path(file_path)
                combo.insertItem(0, f"📄 {p.name}", str(p))
                combo.setCurrentIndex(0)
                self.sound_overrides[row] = p
            else:
                combo.setCurrentIndex(0)
                return
        elif data:
            self.sound_overrides[row] = Path(data)

        self.segment_changed.emit(row)

    def _on_vol_slider_changed(self, row: int, val: int, spin: QDoubleSpinBox) -> None:
        if self._block_updates:
            return
        spin.blockSignals(True)
        spin.setValue(float(val))
        spin.blockSignals(False)
        self.volume_adjustments[row] = float(val)
        self.segment_changed.emit(row)

    def _on_vol_spin_changed(self, row: int, val: float, slider: QSlider) -> None:
        if self._block_updates:
            return
        slider.blockSignals(True)
        slider.setValue(int(val))
        slider.blockSignals(False)
        self.volume_adjustments[row] = val
        self.segment_changed.emit(row)

    def _on_enabled_toggled(self, row: int, enabled: bool) -> None:
        if self._block_updates:
            return
        self.enabled_map[row] = enabled
        self.segment_changed.emit(row)

    def _on_jump_clicked(self, row: int) -> None:
        if 0 <= row < len(self.gaps):
            gap = self.gaps[row]
            pos_ms = int(gap.start_sec * 1000)
            self.seek_requested.emit(pos_ms)

    def _adjust_all_volume(self, delta: float) -> None:
        for row in range(len(self.gaps)):
            cur = self.volume_adjustments.get(row, 0.0)
            new_val = max(-24.0, min(12.0, cur + delta))
            self.volume_adjustments[row] = new_val

            cell = self.table.cellWidget(row, 4)
            if cell:
                spin = cell.findChild(QDoubleSpinBox)
                slider = cell.findChild(QSlider)
                if spin and slider:
                    spin.setValue(new_val)
                    slider.setValue(int(new_val))

        self.apply_requested.emit()

    def _set_all_enabled(self, enabled: bool) -> None:
        for row in range(len(self.gaps)):
            self.enabled_map[row] = enabled
            cell = self.table.cellWidget(row, 5)
            if cell:
                chk = cell.findChild(QCheckBox)
                if chk:
                    chk.setChecked(enabled)
        self.apply_requested.emit()
