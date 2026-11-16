"""Video and Audio Player Widget for ClearSound with Track Switching."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QTime, QUrl, Qt, Signal
from PySide6.QtGui import QIcon, QKeySequence, QShortcut
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from clearsound.core.audio_utils import format_time


class VideoPlayerWidget(QWidget):
    """Media player widget supporting video/audio playback and track mode switching."""

    position_changed = Signal(int)   # ms
    duration_changed = Signal(int)   # ms
    track_mode_changed = Signal(str) # "enhanced", "no_vocals", "original"

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # ── Media Components ──────────────────────────────────────────
        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(1.0)

        # File paths for different tracks
        self.source_path: Optional[Path] = None
        self.is_video: bool = False
        self.track_sources: dict[str, Optional[Path]] = {
            "enhanced": None,
            "no_vocals": None,
            "original": None,
        }
        self.current_mode: str = "enhanced"

        self._setup_ui()
        self._connect_signals()

        # Keyboard shortcut for play/pause
        self.shortcut_space = QShortcut(QKeySequence(Qt.Key_Space), self)
        self.shortcut_space.activated.connect(self.toggle_play)

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(6)

        # ── Viewport Stack (Video or Audio Cover) ──────────────────────
        self.viewport_stack = QStackedWidget(self)
        self.viewport_stack.setMinimumHeight(240)

        # Page 0: Audio Placeholder
        self.audio_placeholder = QFrame()
        self.audio_placeholder.setObjectName("cardPanel")
        audio_layout = QVBoxLayout(self.audio_placeholder)
        audio_layout.setAlignment(Qt.AlignCenter)
        self.audio_icon_label = QLabel("🎵")
        self.audio_icon_label.setStyleSheet("font-size: 48px;")
        self.audio_title_label = QLabel("Файл не загружен")
        self.audio_title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #cbd5e1;")
        self.audio_subtitle_label = QLabel("Загрузите аудио или видео для начала работы")
        self.audio_subtitle_label.setStyleSheet("color: #64748b;")
        audio_layout.addWidget(self.audio_icon_label, alignment=Qt.AlignCenter)
        audio_layout.addWidget(self.audio_title_label, alignment=Qt.AlignCenter)
        audio_layout.addWidget(self.audio_subtitle_label, alignment=Qt.AlignCenter)

        # Page 1: Video Widget
        self.video_widget = QVideoWidget()
        self.video_widget.setStyleSheet("background-color: #0b0d12; border-radius: 6px;")
        self.player.setVideoOutput(self.video_widget)

        self.viewport_stack.addWidget(self.audio_placeholder)
        self.viewport_stack.addWidget(self.video_widget)
        main_layout.addWidget(self.viewport_stack, stretch=1)

        # ── Audio Track Mode Selector Bar ─────────────────────────────
        track_bar = QFrame()
        track_bar.setStyleSheet("background-color: #161a23; border: 1px solid #232c3c; border-radius: 6px; padding: 2px;")
        track_layout = QHBoxLayout(track_bar)
        track_layout.setContentsMargins(8, 4, 8, 4)

        mode_title = QLabel("🎧 Дорожка:")
        mode_title.setStyleSheet("font-weight: bold; color: #94a3b8;")
        track_layout.addWidget(mode_title)

        self.btn_group = QButtonGroup(self)
        self.radio_enhanced = QRadioButton("✨ Восстановленная (с новыми звуками)")
        self.radio_enhanced.setChecked(True)
        self.radio_no_vocals = QRadioButton("🔇 Без голоса (исходный Demucs)")
        self.radio_original = QRadioButton("📼 Исходное аудио/видео")

        self.btn_group.addButton(self.radio_enhanced, 0)
        self.btn_group.addButton(self.radio_no_vocals, 1)
        self.btn_group.addButton(self.radio_original, 2)

        track_layout.addWidget(self.radio_enhanced)
        track_layout.addWidget(self.radio_no_vocals)
        track_layout.addWidget(self.radio_original)
        track_layout.addStretch()

        main_layout.addWidget(track_bar)

        # ── Playback Controls Bar ─────────────────────────────────────
        ctrl_bar = QFrame()
        ctrl_bar.setStyleSheet("background-color: #171b24; border-radius: 6px; padding: 4px;")
        ctrl_layout = QHBoxLayout(ctrl_bar)
        ctrl_layout.setContentsMargins(8, 6, 8, 6)

        # Buttons
        self.btn_rewind = QPushButton("⏪ -5с")
        self.btn_rewind.setToolTip("Назад на 5 секунд (Стрелка влево)")

        self.btn_play = QPushButton("▶ Воспроизвести")
        self.btn_play.setObjectName("primaryButton")
        self.btn_play.setMinimumWidth(130)

        self.btn_forward = QPushButton("⏩ +5с")
        self.btn_forward.setToolTip("Вперед на 5 секунд (Стрелка вправо)")

        ctrl_layout.addWidget(self.btn_rewind)
        ctrl_layout.addWidget(self.btn_play)
        ctrl_layout.addWidget(self.btn_forward)

        # Time Slider
        self.slider_time = QSlider(Qt.Horizontal)
        self.slider_time.setRange(0, 0)
        ctrl_layout.addWidget(self.slider_time, stretch=1)

        # Time Label
        self.label_time = QLabel("00:00.0 / 00:00.0")
        self.label_time.setStyleSheet("font-family: monospace; color: #94a3b8; font-weight: bold;")
        ctrl_layout.addWidget(self.label_time)

        # Volume Controls
        self.btn_mute = QPushButton("🔊")
        self.btn_mute.setFixedWidth(36)
        self.slider_volume = QSlider(Qt.Horizontal)
        self.slider_volume.setRange(0, 100)
        self.slider_volume.setValue(100)
        self.slider_volume.setFixedWidth(80)

        ctrl_layout.addWidget(self.btn_mute)
        ctrl_layout.addWidget(self.slider_volume)

        main_layout.addWidget(ctrl_bar)

    def _connect_signals(self) -> None:
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_rewind.clicked.connect(lambda: self.seek_relative(-5000))
        self.btn_forward.clicked.connect(lambda: self.seek_relative(5000))

        self.btn_group.idClicked.connect(self._on_track_mode_clicked)

        # Player events
        self.player.positionChanged.connect(self._on_position_changed)
        self.player.durationChanged.connect(self._on_duration_changed)
        self.player.playbackStateChanged.connect(self._on_state_changed)

        # Slider interaction
        self.slider_time.sliderMoved.connect(self.set_position)

        # Volume
        self.slider_volume.valueChanged.connect(self._on_volume_changed)
        self.btn_mute.clicked.connect(self._on_mute_clicked)

    # ── Source Setup ──────────────────────────────────────────────────

    def load_media(
        self,
        source_path: Path,
        enhanced_path: Optional[Path] = None,
        no_vocals_path: Optional[Path] = None,
    ) -> None:
        """Load a media file into the player."""
        self.source_path = Path(source_path)
        video_exts = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}
        self.is_video = self.source_path.suffix.lower() in video_exts

        self.track_sources["original"] = self.source_path
        self.track_sources["enhanced"] = enhanced_path or self.source_path
        self.track_sources["no_vocals"] = no_vocals_path or self.source_path

        if self.is_video:
            self.viewport_stack.setCurrentIndex(1)
        else:
            self.viewport_stack.setCurrentIndex(0)
            self.audio_title_label.setText(self.source_path.name)
            self.audio_subtitle_label.setText("Аудиофайл готов к воспроизведению")

        # Load current mode
        self.switch_track(self.current_mode, auto_play=False)

    def update_track_sources(
        self,
        enhanced_path: Optional[Path] = None,
        no_vocals_path: Optional[Path] = None,
    ) -> None:
        """Update paths after separation or enhancement without stopping current position."""
        pos = self.player.position()
        was_playing = self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState

        if enhanced_path:
            self.track_sources["enhanced"] = Path(enhanced_path)
        if no_vocals_path:
            self.track_sources["no_vocals"] = Path(no_vocals_path)

        # Reload currently active track
        self.switch_track(self.current_mode, auto_play=was_playing, resume_pos=pos)

    def switch_track(self, mode: str, auto_play: bool = False, resume_pos: Optional[int] = None) -> None:
        """Switch audio track mode ('enhanced', 'no_vocals', 'original')."""
        self.current_mode = mode
        target = self.track_sources.get(mode) or self.source_path
        if not target or not target.exists():
            return

        if resume_pos is None:
            resume_pos = self.player.position()

        self.player.setSource(QUrl.fromLocalFile(str(target)))
        if resume_pos > 0:
            self.player.setPosition(resume_pos)

        if auto_play:
            self.player.play()

        self.track_mode_changed.emit(mode)

    def _on_track_mode_clicked(self, btn_id: int) -> None:
        modes = {0: "enhanced", 1: "no_vocals", 2: "original"}
        mode = modes.get(btn_id, "enhanced")
        self.switch_track(mode, auto_play=(self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState))

    # ── Playback Controls ─────────────────────────────────────────────

    def toggle_play(self) -> None:
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def play(self) -> None:
        self.player.play()

    def pause(self) -> None:
        self.player.pause()

    def seek_relative(self, delta_ms: int) -> None:
        new_pos = max(0, min(self.player.position() + delta_ms, self.player.duration()))
        self.set_position(new_pos)

    def set_position(self, pos_ms: int) -> None:
        self.player.setPosition(pos_ms)

    def _on_position_changed(self, pos: int) -> None:
        if not self.slider_time.isSliderDown():
            self.slider_time.setValue(pos)
        self._update_time_label(pos, self.player.duration())
        self.position_changed.emit(pos)

    def _on_duration_changed(self, duration: int) -> None:
        self.slider_time.setRange(0, duration)
        self._update_time_label(self.player.position(), duration)
        self.duration_changed.emit(duration)

    def _on_state_changed(self, state: QMediaPlayer.PlaybackState) -> None:
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.btn_play.setText("⏸ Пауза")
        else:
            self.btn_play.setText("▶ Воспроизвести")

    def _update_time_label(self, pos_ms: int, dur_ms: int) -> None:
        cur_str = format_time(pos_ms / 1000.0)
        tot_str = format_time(dur_ms / 1000.0)
        self.label_time.setText(f"{cur_str} / {tot_str}")

    def _on_volume_changed(self, val: int) -> None:
        vol = val / 100.0
        self.audio_output.setVolume(vol)
        if val == 0:
            self.btn_mute.setText("🔇")
        else:
            self.btn_mute.setText("🔊")

    def _on_mute_clicked(self) -> None:
        is_muted = self.audio_output.isMuted()
        self.audio_output.setMuted(not is_muted)
        self.btn_mute.setText("🔇" if not is_muted else "🔊")
