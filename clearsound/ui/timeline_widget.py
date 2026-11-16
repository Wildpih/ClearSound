"""Interactive Timeline and Waveform Display Widget for ClearSound."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from PySide6.QtCore import QPoint, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import QToolTip, QWidget

from clearsound.core.audio_utils import format_time
from clearsound.core.gap_analyzer import GapInfo

CATEGORY_COLORS = {
    "alerts": QColor("#eab308"),
    "donor": QColor("#22c55e"),
    "ambient": QColor("#00d2b4"),
    "nature": QColor("#10b981"),
    "music": QColor("#8b5cf6"),
    "urban": QColor("#f59e0b"),
    "foley": QColor("#ec4899"),
}


class TimelineWidget(QWidget):
    """Interactive timeline showing waveform, detected gaps, and playback playhead."""

    seek_requested = Signal(int)     # position in milliseconds
    gap_selected = Signal(int)       # gap index

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumHeight(110)
        self.setMouseTracking(True)

        self.duration_ms: int = 0
        self.position_ms: int = 0
        self.waveform_peaks: Optional[np.ndarray] = None
        self.gaps: List[GapInfo] = []
        self.enabled_map: dict[int, bool] = {}
        self.selected_gap_idx: int = -1

        self._is_dragging: bool = False

    def set_duration(self, duration_ms: int) -> None:
        self.duration_ms = max(0, duration_ms)
        self.update()

    def set_position(self, position_ms: int) -> None:
        self.position_ms = max(0, min(position_ms, self.duration_ms))
        self.update()

    def set_waveform(self, peaks: Optional[np.ndarray]) -> None:
        self.waveform_peaks = peaks
        self.update()

    def set_gaps(self, gaps: List[GapInfo], enabled_map: Optional[dict[int, bool]] = None) -> None:
        self.gaps = gaps or []
        self.enabled_map = enabled_map or {i: True for i in range(len(self.gaps))}
        self.update()

    def select_gap(self, index: int) -> None:
        if self.selected_gap_idx != index:
            self.selected_gap_idx = index
            self.update()

    # ── Coordinates Conversion ────────────────────────────────────────

    def _ms_to_x(self, ms: int) -> float:
        if self.duration_ms <= 0:
            return 0.0
        return (ms / self.duration_ms) * self.width()

    def _x_to_ms(self, x: float) -> int:
        if self.width() <= 0 or self.duration_ms <= 0:
            return 0
        ratio = max(0.0, min(x / self.width(), 1.0))
        return int(ratio * self.duration_ms)

    def _find_gap_at_x(self, x: float) -> int:
        ms = self._x_to_ms(x)
        sec = ms / 1000.0
        for i, gap in enumerate(self.gaps):
            if gap.start_sec <= sec <= gap.end_sec:
                return i
        return -1

    # ── Paint Event ───────────────────────────────────────────────────

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        w = self.width()
        h = self.height()

        # Background
        painter.fillRect(0, 0, w, h, QColor("#12161f"))

        # Time markers bar at the top
        timebar_h = 22
        painter.fillRect(0, 0, w, timebar_h, QColor("#181d28"))
        painter.setPen(QColor("#273244"))
        painter.drawLine(0, timebar_h, w, timebar_h)

        # Draw time markers
        if self.duration_ms > 0:
            font = QFont("Segoe UI", 9)
            painter.setFont(font)
            painter.setPen(QColor("#64748b"))

            # Determine marker interval based on duration
            total_sec = self.duration_ms / 1000.0
            if total_sec < 30:
                interval_sec = 5
            elif total_sec < 120:
                interval_sec = 10
            elif total_sec < 600:
                interval_sec = 30
            else:
                interval_sec = 60

            t = 0
            while t <= total_sec:
                x = self._ms_to_x(int(t * 1000))
                painter.drawLine(int(x), 0, int(x), timebar_h)
                painter.drawText(int(x) + 4, 15, format_time(t))
                t += interval_sec

        # Waveform area
        wave_top = timebar_h + 4
        wave_h = h - wave_top - 4
        center_y = wave_top + (wave_h / 2)

        if self.waveform_peaks is not None and len(self.waveform_peaks) > 0:
            num_peaks = len(self.waveform_peaks)
            painter.setPen(QPen(QColor("#2d3b50"), 1))
            for i in range(num_peaks):
                x = (i / num_peaks) * w
                min_val = self.waveform_peaks[i, 0]
                max_val = self.waveform_peaks[i, 1]
                y1 = center_y - (max_val * (wave_h / 2) * 0.9)
                y2 = center_y - (min_val * (wave_h / 2) * 0.9)
                painter.drawLine(int(x), int(y1), int(x), int(y2))
        else:
            # Baseline
            painter.setPen(QPen(QColor("#1e2533"), 1, Qt.DashLine))
            painter.drawLine(0, int(center_y), w, int(center_y))

        # ── Draw Gaps ─────────────────────────────────────────────────
        gap_font = QFont("Segoe UI", 8, QFont.Bold)
        painter.setFont(gap_font)

        for i, gap in enumerate(self.gaps):
            start_x = self._ms_to_x(int(gap.start_sec * 1000))
            end_x = self._ms_to_x(int(gap.end_sec * 1000))
            gap_w = max(4.0, end_x - start_x)

            color = CATEGORY_COLORS.get(gap.category_hint.lower(), QColor("#00d2b4"))
            is_enabled = self.enabled_map.get(i, True)
            is_selected = (i == self.selected_gap_idx)

            gap_rect = QRectF(start_x, wave_top, gap_w, wave_h)

            if is_enabled:
                fill_color = QColor(color)
                fill_color.setAlpha(85 if not is_selected else 140)
                border_color = QColor(color)
            else:
                fill_color = QColor("#334155")
                fill_color.setAlpha(60)
                border_color = QColor("#64748b")

            painter.setBrush(QBrush(fill_color))
            pen = QPen(border_color, 2 if is_selected else 1)
            painter.setPen(pen)
            painter.drawRoundedRect(gap_rect, 4, 4)

            # Draw tag text if wide enough
            if gap_w > 40:
                painter.setPen(QColor("#ffffff" if is_enabled else "#94a3b8"))
                label = f"#{i+1} {gap.category_hint[:4]}"
                painter.drawText(gap_rect.adjusted(4, 2, -4, -2), Qt.AlignLeft | Qt.AlignTop, label)

        # ── Draw Playhead ─────────────────────────────────────────────
        if self.duration_ms > 0:
            playhead_x = self._ms_to_x(self.position_ms)
            # Red/cyan glowing line
            painter.setPen(QPen(QColor("#00f0cf"), 2))
            painter.drawLine(int(playhead_x), 0, int(playhead_x), h)

            # Playhead triangle head at the time bar
            painter.setBrush(QBrush(QColor("#00f0cf")))
            painter.setPen(Qt.NoPen)
            poly = [
                QPoint(int(playhead_x) - 5, 0),
                QPoint(int(playhead_x) + 5, 0),
                QPoint(int(playhead_x), 9),
            ]
            painter.drawPolygon(poly)

    # ── Mouse Interaction ─────────────────────────────────────────────

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self._is_dragging = True
            pos_ms = self._x_to_ms(event.position().x())
            self.seek_requested.emit(pos_ms)

            # Check if clicked on a gap
            gap_idx = self._find_gap_at_x(event.position().x())
            if gap_idx != -1:
                self.selected_gap_idx = gap_idx
                self.gap_selected.emit(gap_idx)
                self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        x = event.position().x()
        if self._is_dragging:
            pos_ms = self._x_to_ms(x)
            self.seek_requested.emit(pos_ms)
        else:
            # Tooltip
            ms = self._x_to_ms(x)
            gap_idx = self._find_gap_at_x(x)
            if gap_idx != -1:
                gap = self.gaps[gap_idx]
                tooltip_text = (
                    f"Сегмент #{gap_idx+1} ({gap.category_hint})\n"
                    f"Время: {format_time(gap.start_sec)} - {format_time(gap.end_sec)} "
                    f"({gap.duration_sec:.2f}с)\n"
                    f"Потеря энергии: {gap.energy_loss_ratio:.0%}"
                )
            else:
                tooltip_text = f"Время: {format_time(ms / 1000.0)}"
            QToolTip.showText(event.globalPosition().toPoint(), tooltip_text, self)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self._is_dragging = False
