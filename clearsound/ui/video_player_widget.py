"""Video and Audio Player Widget for ClearSound with Track Switching and Audio Device Routing."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from PySide6.QtCore import QTime, QUrl, Qt, Signal
from PySide6.QtGui import QIcon, QKeySequence, QShortcut
from PySide6.QtMultimedia import QAudioDevice, QAudioOutput, QMediaDevices, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
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
    """Media player widget supporting video/audio playback, audio device selection, and track switching."""

    position_changed = Signal(int)   # ms
    duration_changed = Signal(int)   # ms
    track_mode_changed = Signal(str) # "enhanced", "no_vocals", "original"
    device_changed = Signal(str)     # device description

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # ── Media Components ──────────────────────────────────────────
        self.media_devices = QMediaDevices(self)
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
        self._test_sound_player: Optional[QMediaPlayer] = None

        self._setup_ui()
        self._init_audio_devices()
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
        self.viewport_stack.setMinimumHeight(260)

        # Page 0: Audio Placeholder
        self.audio_placeholder = QFrame()
        self.audio_placeholder.setObjectName("cardPanel")
        audio_layout = QVBoxLayout(self.audio_placeholder)
        audio_layout.setAlignment(Qt.AlignCenter)
        audio_layout.setSpacing(8)

        self.audio_icon_label = QLabel("🎵")
        self.audio_icon_label.setStyleSheet("font-size: 52px;")
        self.audio_title_label = QLabel("Файл не загружен")
        self.audio_title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #ffffff;")
        self.audio_subtitle_label = QLabel("Загрузите видеоролик или аудиофайл для начала работы")
        self.audio_subtitle_label.setStyleSheet("color: #94a3b8; font-size: 13px;")

        audio_layout.addWidget(self.audio_icon_label, alignment=Qt.AlignCenter)
        audio_layout.addWidget(self.audio_title_label, alignment=Qt.AlignCenter)
        audio_layout.addWidget(self.audio_subtitle_label, alignment=Qt.AlignCenter)

        # Page 1: Video Widget
        self.video_widget = QVideoWidget()
        self.video_widget.setStyleSheet("background-color: #07090e; border-radius: 8px;")
        self.player.setVideoOutput(self.video_widget)

        self.viewport_stack.addWidget(self.audio_placeholder)
        self.viewport_stack.addWidget(self.video_widget)
        main_layout.addWidget(self.viewport_stack, stretch=1)

        # ── Audio Track Mode Selector Bar ─────────────────────────────
        track_bar = QFrame()
        track_bar.setStyleSheet(
            "background-color: #141822; border: 1px solid #283244; "
            "border-radius: 8px; padding: 4px;"
        )
        track_layout = QHBoxLayout(track_bar)
        track_layout.setContentsMargins(12, 6, 12, 6)
        track_layout.setSpacing(16)

        mode_title = QLabel("🎧 Режим звука:")
        mode_title.setStyleSheet("font-weight: 700; color: #00e5c5; font-size: 13px;")
        track_layout.addWidget(mode_title)

        self.btn_group = QButtonGroup(self)
        self.radio_enhanced = QRadioButton("✨ Восстановленная (с новыми звуками)")
        self.radio_enhanced.setChecked(True)
        self.radio_enhanced.setStyleSheet("font-weight: 600; color: #ffffff;")

        self.radio_no_vocals = QRadioButton("🔇 Без голоса (Demucs)")
        self.radio_no_vocals.setStyleSheet("font-weight: 600; color: #ffffff;")

        self.radio_original = QRadioButton("📼 Исходное аудио/видео")
        self.radio_original.setStyleSheet("font-weight: 600; color: #ffffff;")

        self.btn_group.addButton(self.radio_enhanced, 0)
        self.btn_group.addButton(self.radio_no_vocals, 1)
        self.btn_group.addButton(self.radio_original, 2)

        track_layout.addWidget(self.radio_enhanced)
        track_layout.addWidget(self.radio_no_vocals)
        track_layout.addWidget(self.radio_original)
        track_layout.addStretch()

        main_layout.addWidget(track_bar)

        # ── Audio Output Device Bar ───────────────────────────────────
        device_bar = QFrame()
        device_bar.setStyleSheet(
            "background-color: #161b26; border: 1px solid #2a3447; "
            "border-radius: 6px; padding: 2px;"
        )
        dev_layout = QHBoxLayout(device_bar)
        dev_layout.setContentsMargins(10, 4, 10, 4)
        dev_layout.setSpacing(10)

        lbl_device = QLabel("🔊 Устройство вывода:")
        lbl_device.setStyleSheet("font-weight: 600; color: #cbd5e1; font-size: 12px;")
        dev_layout.addWidget(lbl_device)

        self.combo_audio_devices = QComboBox()
        self.combo_audio_devices.setMinimumWidth(260)
        self.combo_audio_devices.setToolTip("Выберите наушники или динамики для воспроизведения звука")
        dev_layout.addWidget(self.combo_audio_devices)

        self.btn_test_audio = QPushButton("🔔 Проверить звук в наушниках")
        self.btn_test_audio.setToolTip("Воспроизвести короткий звуковой сигнал для проверки наушников")
        self.btn_test_audio.setStyleSheet(
            "background-color: #1e293b; color: #00f0cf; border: 1px solid #00d2b4; "
            "font-weight: 600; padding: 4px 12px; border-radius: 4px;"
        )
        self.btn_test_audio.clicked.connect(self._on_test_audio_clicked)
        dev_layout.addWidget(self.btn_test_audio)

        self.lbl_audio_test_status = QLabel("")
        self.lbl_audio_test_status.setStyleSheet("color: #10b981; font-weight: 600; font-size: 12px;")
        dev_layout.addWidget(self.lbl_audio_test_status)

        dev_layout.addStretch()
        main_layout.addWidget(device_bar)

        # ── Playback Controls Bar ─────────────────────────────────────
        ctrl_bar = QFrame()
        ctrl_bar.setStyleSheet(
            "background-color: #141822; border: 1px solid #283244; "
            "border-radius: 8px; padding: 6px;"
        )
        ctrl_layout = QHBoxLayout(ctrl_bar)
        ctrl_layout.setContentsMargins(10, 8, 10, 8)
        ctrl_layout.setSpacing(10)

        # Buttons
        self.btn_rewind = QPushButton("⏪ -5с")
        self.btn_rewind.setToolTip("Назад на 5 секунд (Стрелка влево)")
        self.btn_rewind.setStyleSheet("font-weight: bold; min-width: 60px; padding: 8px 12px;")

        self.btn_play = QPushButton("▶ Воспроизвести")
        self.btn_play.setObjectName("primaryButton")
        self.btn_play.setMinimumWidth(150)
        self.btn_play.setStyleSheet("font-size: 14px; font-weight: bold; padding: 8px 20px;")

        self.btn_forward = QPushButton("⏩ +5с")
        self.btn_forward.setToolTip("Вперед на 5 секунд (Стрелка вправо)")
        self.btn_forward.setStyleSheet("font-weight: bold; min-width: 60px; padding: 8px 12px;")

        ctrl_layout.addWidget(self.btn_rewind)
        ctrl_layout.addWidget(self.btn_play)
        ctrl_layout.addWidget(self.btn_forward)

        # Time Slider
        self.slider_time = QSlider(Qt.Horizontal)
        self.slider_time.setRange(0, 0)
        self.slider_time.setCursor(Qt.PointingHandCursor)
        ctrl_layout.addWidget(self.slider_time, stretch=1)

        # Time Label
        self.label_time = QLabel("00:00.0 / 00:00.0")
        self.label_time.setStyleSheet(
            "font-family: 'Consolas', 'Courier New', monospace; "
            "color: #00f0cf; font-weight: bold; font-size: 13px;"
        )
        ctrl_layout.addWidget(self.label_time)

        # Volume Controls
        self.btn_mute = QPushButton("🔊")
        self.btn_mute.setFixedWidth(40)
        self.btn_mute.setStyleSheet("font-size: 15px; padding: 6px;")

        self.slider_volume = QSlider(Qt.Horizontal)
        self.slider_volume.setRange(0, 150)  # Up to 150% volume boost
        self.slider_volume.setValue(100)
        self.slider_volume.setFixedWidth(90)
        self.slider_volume.setToolTip("Громкость воспроизведения (до 150%)")

        self.label_volume_pct = QLabel("100%")
        self.label_volume_pct.setStyleSheet("font-size: 11px; font-weight: bold; color: #cbd5e1; min-width: 38px;")

        ctrl_layout.addWidget(self.btn_mute)
        ctrl_layout.addWidget(self.slider_volume)
        ctrl_layout.addWidget(self.label_volume_pct)

        main_layout.addWidget(ctrl_bar)

    def _init_audio_devices(self) -> None:
        """Populate the audio output device list."""
        self.combo_audio_devices.blockSignals(True)
        self.combo_audio_devices.clear()

        devices = QMediaDevices.audioOutputs()
        default_dev = QMediaDevices.defaultAudioOutput()
        selected_idx = 0

        for i, dev in enumerate(devices):
            desc = dev.description()
            icon_prefix = "🎧 " if any(w in desc.lower() for w in ["наушники", "headphone", "buds", "airpod", "headset"]) else "🔊 "
            is_def = (dev.id() == default_dev.id())
            label = f"{icon_prefix}{desc}" + (" (По умолчанию)" if is_def else "")
            self.combo_audio_devices.addItem(label, dev)
            if is_def:
                selected_idx = i

        self.combo_audio_devices.setCurrentIndex(selected_idx)
        self.combo_audio_devices.blockSignals(False)

        # Set default device
        if devices:
            self.audio_output.setDevice(devices[selected_idx])

    def _connect_signals(self) -> None:
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_rewind.clicked.connect(lambda: self.seek_relative(-5000))
        self.btn_forward.clicked.connect(lambda: self.seek_relative(5000))

        self.btn_group.idClicked.connect(self._on_track_mode_clicked)

        # Audio device change
        self.combo_audio_devices.currentIndexChanged.connect(self._on_audio_device_changed)
        self.media_devices.audioOutputsChanged.connect(self._init_audio_devices)

        # Player events
        self.player.positionChanged.connect(self._on_position_changed)
        self.player.durationChanged.connect(self._on_duration_changed)
        self.player.playbackStateChanged.connect(self._on_state_changed)

        # Slider interaction
        self.slider_time.sliderMoved.connect(self.set_position)

        # Volume
        self.slider_volume.valueChanged.connect(self._on_volume_changed)
        self.btn_mute.clicked.connect(self._on_mute_clicked)

    def _on_audio_device_changed(self, idx: int) -> None:
        dev = self.combo_audio_devices.itemData(idx)
        if isinstance(dev, QAudioDevice):
            self.audio_output.setDevice(dev)
            self.device_changed.emit(dev.description())
            self.lbl_audio_test_status.setText(f"Активно: {dev.description()[:25]}")

    def _on_test_audio_clicked(self) -> None:
        """Generate and play a crisp, pleasant test chime to verify headphones audio."""
        sr = 44100
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        # Harmonious dual tone (A4 440Hz + E5 660Hz) with smooth decay envelope
        env = np.exp(-t * 6.0)
        signal = (0.5 * np.sin(2 * np.pi * 523.25 * t) + 0.3 * np.sin(2 * np.pi * 659.25 * t)) * env
        signal = np.clip(signal, -1.0, 1.0)

        temp_test_wav = Path(tempfile.gettempdir()) / "clearsound_audio_test.wav"
        sf.write(str(temp_test_wav), signal, sr)

        if self._test_sound_player is None:
            self._test_sound_player = QMediaPlayer(self)
        self._test_sound_player.setAudioOutput(self.audio_output)
        self._test_sound_player.setSource(QUrl.fromLocalFile(str(temp_test_wav)))
        self._test_sound_player.play()

        active_dev = self.audio_output.device().description()
        self.lbl_audio_test_status.setText(f"🔔 Звук отправлен на: {active_dev}!")

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
        """Update paths after separation or enhancement without losing current timestamp."""
        pos = self.player.position()
        was_playing = self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState

        if enhanced_path:
            self.track_sources["enhanced"] = Path(enhanced_path)
        if no_vocals_path:
            self.track_sources["no_vocals"] = Path(no_vocals_path)

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
            self.btn_play.setStyleSheet(
                "background-color: #10b981; color: #ffffff; border: 1px solid #34d399; "
                "font-size: 14px; font-weight: bold; padding: 8px 20px; border-radius: 6px;"
            )
        else:
            self.btn_play.setText("▶ Воспроизвести")
            self.btn_play.setStyleSheet(
                "background-color: #00d2b4; color: #071118; border: 1px solid #00f0cf; "
                "font-size: 14px; font-weight: bold; padding: 8px 20px; border-radius: 6px;"
            )

    def _update_time_label(self, pos_ms: int, dur_ms: int) -> None:
        cur_str = format_time(pos_ms / 1000.0)
        tot_str = format_time(dur_ms / 1000.0)
        self.label_time.setText(f"{cur_str} / {tot_str}")

    def _on_volume_changed(self, val: int) -> None:
        # val 0 to 150
        vol = val / 100.0
        self.audio_output.setVolume(min(1.0, vol))
        self.label_volume_pct.setText(f"{val}%")
        if val == 0:
            self.btn_mute.setText("🔇")
        else:
            self.btn_mute.setText("🔊")

    def _on_mute_clicked(self) -> None:
        is_muted = self.audio_output.isMuted()
        self.audio_output.setMuted(not is_muted)
        self.btn_mute.setText("🔇" if not is_muted else "🔊")
