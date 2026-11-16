"""High-contrast, professional DAW-grade Dark Theme for ClearSound GUI."""

DARK_THEME_QSS = """
/* ── Global Base ─────────────────────────────────────────── */
QWidget {
    background-color: #0d1117;
    color: #ffffff;
    font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, "Segoe UI Variable", Roboto, sans-serif;
    font-size: 13px;
    selection-background-color: #00d2b4;
    selection-color: #0d1117;
}

QMainWindow, QDialog {
    background-color: #0d1117;
}

/* ── Panels & GroupBoxes ─────────────────────────────────── */
QFrame#cardPanel, QGroupBox {
    background-color: #161b24;
    border: 1px solid #2d3748;
    border-radius: 8px;
    margin-top: 10px;
    padding: 12px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 2px 10px;
    color: #00e5c5;
    font-weight: bold;
    font-size: 13px;
    background-color: #161b24;
}

/* ── Push Buttons (Standard) ─────────────────────────────── */
QPushButton {
    background-color: #212836;
    color: #ffffff;
    border: 1px solid #3d4a61;
    border-radius: 6px;
    padding: 7px 16px;
    font-size: 12px;
    font-weight: 600;
}

QPushButton:hover {
    background-color: #2b3547;
    border-color: #00e5c5;
    color: #ffffff;
}

QPushButton:pressed {
    background-color: #181f2c;
    border-color: #00b89e;
}

QPushButton:disabled {
    background-color: #161a22;
    color: #718096;
    border-color: #252d3d;
}

/* ── Push Buttons (Primary / Teal) ───────────────────────── */
QPushButton#primaryButton {
    background-color: #00d2b4;
    color: #071118;
    border: 1px solid #00f0cf;
    border-radius: 6px;
    font-weight: 700;
    font-size: 13px;
    padding: 8px 20px;
}

QPushButton#primaryButton:hover {
    background-color: #1af2d4;
    border-color: #ffffff;
    color: #000000;
}

QPushButton#primaryButton:pressed {
    background-color: #00b399;
}

QPushButton#primaryButton:disabled {
    background-color: #133a36;
    color: #4a7772;
    border-color: #1a4d47;
}

/* ── Push Buttons (Accent / Violet) ──────────────────────── */
QPushButton#accentButton {
    background-color: #8b5cf6;
    color: #ffffff;
    border: 1px solid #a78bfa;
    border-radius: 6px;
    font-weight: 700;
    font-size: 13px;
    padding: 8px 20px;
}

QPushButton#accentButton:hover {
    background-color: #9d74f7;
    border-color: #c4b5fd;
}

QPushButton#accentButton:pressed {
    background-color: #7c3aed;
}

QPushButton#accentButton:disabled {
    background-color: #2a2040;
    color: #645582;
    border-color: #3b2e59;
}

/* ── Table Action Buttons ────────────────────────────────── */
QPushButton.tableActionBtn {
    background-color: #242d3d;
    color: #ffffff;
    border: 1px solid #414f68;
    border-radius: 4px;
    padding: 3px 8px;
    font-size: 11px;
    font-weight: 600;
    min-height: 24px;
}

QPushButton.tableActionBtn:hover {
    background-color: #313d52;
    border-color: #00e5c5;
    color: #00f0cf;
}

QPushButton.tableActionBtn:pressed {
    background-color: #1a202c;
}

/* ── Radio Buttons ───────────────────────────────────────── */
QRadioButton {
    color: #f1f5f9;
    font-weight: 500;
    font-size: 13px;
    spacing: 8px;
    padding: 4px;
}

QRadioButton::indicator {
    width: 18px;
    height: 18px;
    border-radius: 9px;
    border: 2px solid #475569;
    background-color: #161b24;
}

QRadioButton::indicator:hover {
    border-color: #00d2b4;
    background-color: #1e2636;
}

QRadioButton::indicator:checked {
    border: 2px solid #00d2b4;
    background-color: #00d2b4;
}

/* ── Check Boxes ─────────────────────────────────────────── */
QCheckBox {
    color: #f1f5f9;
    font-weight: 500;
    spacing: 8px;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 2px solid #475569;
    background-color: #161b24;
}

QCheckBox::indicator:hover {
    border-color: #00d2b4;
    background-color: #1e2636;
}

QCheckBox::indicator:checked {
    border: 2px solid #00d2b4;
    background-color: #00d2b4;
}

/* ── Sliders ─────────────────────────────────────────────── */
QSlider::groove:horizontal {
    height: 6px;
    background: #252d3d;
    border-radius: 3px;
}

QSlider::sub-page:horizontal {
    background: #00d2b4;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #ffffff;
    border: 2px solid #00d2b4;
    width: 16px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 8px;
}

QSlider::handle:horizontal:hover {
    background: #00f0cf;
    border-color: #ffffff;
}

/* ── Inputs & Combo Boxes ────────────────────────────────── */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #161b24;
    color: #ffffff;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 5px 10px;
    font-size: 12px;
    min-height: 22px;
}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid #00d2b4;
    background-color: #1a2230;
}

QComboBox::drop-down {
    border: none;
    width: 24px;
    background: transparent;
}

QComboBox::down-arrow {
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 5px solid #94a3b8;
    margin-right: 6px;
}

QComboBox QAbstractItemView {
    background-color: #161b24;
    color: #ffffff;
    border: 1px solid #334155;
    selection-background-color: #00d2b4;
    selection-color: #071118;
    padding: 4px;
}

/* ── Tab Widget ──────────────────────────────────────────── */
QTabWidget::pane {
    border: 1px solid #2d3748;
    background-color: #131720;
    border-radius: 6px;
    top: -1px;
}

QTabBar::tab {
    background-color: #1a202c;
    color: #94a3b8;
    padding: 10px 22px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    border: 1px solid #2d3748;
    border-bottom: none;
    margin-right: 4px;
    font-weight: 600;
    font-size: 13px;
}

QTabBar::tab:selected {
    background-color: #131720;
    color: #00f0cf;
    border-top: 3px solid #00d2b4;
    font-weight: 700;
}

QTabBar::tab:hover:!selected {
    background-color: #242c3d;
    color: #ffffff;
}

/* ── Scrollbars ──────────────────────────────────────────── */
QScrollBar:vertical {
    border: none;
    background: #11141b;
    width: 10px;
    border-radius: 5px;
}

QScrollBar::handle:vertical {
    background: #334155;
    min-height: 25px;
    border-radius: 5px;
}

QScrollBar::handle:vertical:hover {
    background: #475569;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    border: none;
    background: #11141b;
    height: 10px;
    border-radius: 5px;
}

QScrollBar::handle:horizontal {
    background: #334155;
    min-width: 25px;
    border-radius: 5px;
}

QScrollBar::handle:horizontal:hover {
    background: #475569;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

/* ── Tables & Trees ──────────────────────────────────────── */
QTableWidget, QTreeWidget, QListWidget {
    background-color: #141820;
    border: 1px solid #2d3748;
    border-radius: 6px;
    gridline-color: #212836;
    color: #ffffff;
    alternate-background-color: #171c26;
}

QTableWidget::item, QListWidget::item {
    padding: 6px 10px;
}

QTableWidget::item:selected, QListWidget::item:selected {
    background-color: #1e3a47;
    color: #00f0cf;
    border-left: 3px solid #00d2b4;
}

QHeaderView::section {
    background-color: #1c2331;
    color: #cbd5e1;
    padding: 8px 10px;
    border: none;
    border-right: 1px solid #2d3748;
    border-bottom: 1px solid #2d3748;
    font-weight: 700;
    font-size: 12px;
}

/* ── Progress Bar ────────────────────────────────────────── */
QProgressBar {
    border: 1px solid #334155;
    border-radius: 6px;
    text-align: center;
    background-color: #141820;
    color: #ffffff;
    font-weight: bold;
    height: 20px;
}

QProgressBar::chunk {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00d2b4, stop:1 #8b5cf6);
    border-radius: 5px;
}

/* ── Status Bar ──────────────────────────────────────────── */
QStatusBar {
    background-color: #0a0d13;
    color: #94a3b8;
    border-top: 1px solid #1e2533;
}
"""
