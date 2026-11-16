"""Modern Dark UI Theme and Style Sheets for ClearSound GUI."""

DARK_THEME_QSS = """
/* ── Global Base ─────────────────────────────────────────── */
QWidget {
    background-color: #12151b;
    color: #e2e8f0;
    font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    font-size: 13px;
    selection-background-color: #00d2b4;
    selection-color: #12151b;
}

QMainWindow, QDialog {
    background-color: #12151b;
}

/* ── Panels & GroupBoxes ─────────────────────────────────── */
QFrame#cardPanel, QGroupBox {
    background-color: #1a1e27;
    border: 1px solid #28303f;
    border-radius: 8px;
    margin-top: 10px;
    padding: 12px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 2px 8px;
    color: #00d2b4;
    font-weight: bold;
    font-size: 13px;
}

/* ── Push Buttons ────────────────────────────────────────── */
QPushButton {
    background-color: #242c3b;
    color: #f1f5f9;
    border: 1px solid #364255;
    border-radius: 6px;
    padding: 6px 14px;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #2e384b;
    border-color: #00d2b4;
    color: #ffffff;
}

QPushButton:pressed {
    background-color: #1b222f;
}

QPushButton:disabled {
    background-color: #181c25;
    color: #556277;
    border-color: #222937;
}

QPushButton#primaryButton {
    background-color: #00b89e;
    color: #091316;
    border: 1px solid #00d2b4;
    font-weight: bold;
    padding: 7px 18px;
}

QPushButton#primaryButton:hover {
    background-color: #00e5c5;
    border-color: #5df7e0;
}

QPushButton#primaryButton:pressed {
    background-color: #009681;
}

QPushButton#accentButton {
    background-color: #7c3aed;
    color: #ffffff;
    border: 1px solid #9061f9;
    font-weight: bold;
    padding: 7px 18px;
}

QPushButton#accentButton:hover {
    background-color: #8b5cf6;
    border-color: #a78bfa;
}

/* ── ToolBar ─────────────────────────────────────────────── */
QToolBar {
    background-color: #171b23;
    border-bottom: 1px solid #262e3d;
    padding: 4px;
    spacing: 6px;
}

QToolButton {
    background-color: transparent;
    color: #cbd5e1;
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 6px 10px;
    font-weight: 500;
}

QToolButton:hover {
    background-color: #242c3b;
    border-color: #3b475c;
    color: #ffffff;
}

QToolButton:pressed, QToolButton:checked {
    background-color: #2c3647;
    border-color: #00d2b4;
    color: #00d2b4;
}

/* ── Sliders ─────────────────────────────────────────────── */
QSlider::groove:horizontal {
    height: 6px;
    background: #252d3c;
    border-radius: 3px;
}

QSlider::sub-page:horizontal {
    background: #00d2b4;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #f8fafc;
    border: 2px solid #00d2b4;
    width: 14px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 7px;
}

QSlider::handle:horizontal:hover {
    background: #00d2b4;
    border-color: #ffffff;
}

/* ── Inputs & Combo ──────────────────────────────────────── */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #171b23;
    color: #f1f5f9;
    border: 1px solid #2e3748;
    border-radius: 6px;
    padding: 5px 8px;
}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid #00d2b4;
    background-color: #1a202c;
}

QComboBox::drop-down {
    border: none;
    width: 20px;
}

QComboBox QAbstractItemView {
    background-color: #1a202c;
    border: 1px solid #333f52;
    selection-background-color: #00d2b4;
    selection-color: #12151b;
    padding: 4px;
}

/* ── Tab Widget ──────────────────────────────────────────── */
QTabWidget::pane {
    border: 1px solid #27303f;
    background-color: #171c24;
    border-radius: 6px;
}

QTabBar::tab {
    background-color: #1a202c;
    color: #94a3b8;
    padding: 8px 18px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    border: 1px solid #283242;
    border-bottom: none;
    margin-right: 3px;
    font-weight: 500;
}

QTabBar::tab:selected {
    background-color: #171c24;
    color: #00d2b4;
    border-top: 2px solid #00d2b4;
    font-weight: bold;
}

QTabBar::tab:hover:!selected {
    background-color: #242c3c;
    color: #ffffff;
}

/* ── Scrollbars ──────────────────────────────────────────── */
QScrollBar:vertical {
    border: none;
    background: #14171e;
    width: 10px;
    border-radius: 5px;
}

QScrollBar::handle:vertical {
    background: #2e3748;
    min-height: 25px;
    border-radius: 5px;
}

QScrollBar::handle:vertical:hover {
    background: #3f4b61;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    border: none;
    background: #14171e;
    height: 10px;
    border-radius: 5px;
}

QScrollBar::handle:horizontal {
    background: #2e3748;
    min-width: 25px;
    border-radius: 5px;
}

QScrollBar::handle:horizontal:hover {
    background: #3f4b61;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

/* ── Tables & Trees ──────────────────────────────────────── */
QTableWidget, QTreeWidget, QListWidget {
    background-color: #161a22;
    border: 1px solid #283242;
    border-radius: 6px;
    gridline-color: #232c3a;
    color: #e2e8f0;
}

QTableWidget::item, QListWidget::item {
    padding: 6px;
}

QTableWidget::item:selected, QListWidget::item:selected {
    background-color: #223240;
    color: #00f0cf;
    border-left: 3px solid #00d2b4;
}

QHeaderView::section {
    background-color: #1e2430;
    color: #94a3b8;
    padding: 6px;
    border: none;
    border-right: 1px solid #283242;
    border-bottom: 1px solid #283242;
    font-weight: bold;
    font-size: 12px;
}

/* ── Progress Bar ────────────────────────────────────────── */
QProgressBar {
    border: 1px solid #2d3748;
    border-radius: 6px;
    text-align: center;
    background-color: #151921;
    color: #ffffff;
    font-weight: bold;
    height: 18px;
}

QProgressBar::chunk {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00b89e, stop:1 #8b5cf6);
    border-radius: 5px;
}

/* ── Status Bar ──────────────────────────────────────────── */
QStatusBar {
    background-color: #0e1116;
    color: #7e8c9f;
    border-top: 1px solid #202733;
}
"""
