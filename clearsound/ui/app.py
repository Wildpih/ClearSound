"""Application entry point for ClearSound GUI."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from clearsound.ui.main_window import MainWindow
from clearsound.ui.theme import DARK_THEME_QSS


def run_app(file_to_open: Optional[str] = None) -> int:
    """Launch the ClearSound graphical interface."""
    app = QApplication(sys.argv)
    app.setApplicationName("ClearSound")
    app.setOrganizationName("ClearSound")

    # Set default modern UI font
    app.setFont(QFont("Segoe UI", 10))

    # Apply dark theme
    app.setStyleSheet(DARK_THEME_QSS)

    # Check CLI argument for initial file if not explicitly given
    if file_to_open is None and len(sys.argv) > 1:
        potential_file = sys.argv[1]
        if Path(potential_file).exists():
            file_to_open = potential_file

    window = MainWindow(initial_file=file_to_open)
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(run_app())
