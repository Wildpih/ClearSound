"""Local sound library management.

Scans the sounds/ directory for audio files, categorizes them based
on subdirectory structure, and provides search/filter functionality.
Supports indexing spectral features for fast matching.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from rich.console import Console
from rich.table import Table

from clearsound.config import Config

console = Console()

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".aac", ".m4a"}


@dataclass
class SoundEntry:
    """A single sound in the library."""

    path: Path
    name: str
    category: str        # subdirectory name: ambient, music, nature, urban, foley
    duration_sec: float = 0.0
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "name": self.name,
            "category": self.category,
            "duration_sec": self.duration_sec,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SoundEntry":
        return cls(
            path=Path(d["path"]),
            name=d["name"],
            category=d["category"],
            duration_sec=d.get("duration_sec", 0.0),
            tags=d.get("tags", []),
        )


class SoundLibrary:
    """Manages the local collection of replacement sounds."""

    KNOWN_CATEGORIES = {"alerts", "foley", "ambient", "music", "nature", "urban", "donor"}
    INDEX_FILE = "library_index.json"

    def __init__(self, config: Config):
        self.config = config
        self.sounds_dir = config.sounds_dir
        self._entries: List[SoundEntry] = []
        self._index_path = self.sounds_dir / self.INDEX_FILE

    # ── Scanning & Indexing ───────────────────────────────────────────

    def scan(self) -> int:
        """Scan the sounds directory and index all audio files.

        Returns:
            Number of sounds found.
        """
        self._entries.clear()

        for audio_file in self._find_audio_files():
            # Category = parent directory name
            category = audio_file.parent.name
            if category not in self.KNOWN_CATEGORIES:
                category = "foley"  # default

            entry = SoundEntry(
                path=audio_file,
                name=audio_file.stem,
                category=category,
            )

            # Try to get duration
            try:
                import soundfile as sf
                info = sf.info(str(audio_file))
                entry.duration_sec = info.duration
            except Exception:
                pass

            # Infer tags from filename
            entry.tags = self._tags_from_name(audio_file.stem)

            self._entries.append(entry)

        # Save index
        self._save_index()

        return len(self._entries)

    def _find_audio_files(self) -> List[Path]:
        """Recursively find audio files in the sounds directory."""
        files = []
        for ext in AUDIO_EXTENSIONS:
            files.extend(self.sounds_dir.rglob(f"*{ext}"))
        return sorted(files)

    def _tags_from_name(self, name: str) -> List[str]:
        """Extract tags from the filename by splitting on common separators."""
        # Replace common separators with spaces
        for sep in ["_", "-", ".", "(", ")", "[", "]"]:
            name = name.replace(sep, " ")
        words = [w.lower().strip() for w in name.split() if len(w) > 2]
        return words

    def _save_index(self) -> None:
        """Save the index to a JSON file."""
        data = [e.to_dict() for e in self._entries]
        try:
            self._index_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            try:
                alt_path = self.config.temp_dir / self.INDEX_FILE
                alt_path.write_text(
                    json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                self._index_path = alt_path
            except Exception:
                pass

    def load_index(self) -> bool:
        """Load a previously saved index.

        Returns:
            True if index was loaded successfully.
        """
        if not self._index_path.exists():
            alt = self.config.temp_dir / self.INDEX_FILE
            if alt.exists():
                self._index_path = alt
            else:
                return False
        try:
            data = json.loads(self._index_path.read_text(encoding="utf-8"))
            self._entries = [SoundEntry.from_dict(d) for d in data]
            # Verify paths still exist
            self._entries = [e for e in self._entries if e.path.exists()]
            return len(self._entries) > 0
        except Exception:
            return False

    # ── Querying ──────────────────────────────────────────────────────

    def list_all(self) -> List[SoundEntry]:
        """Return all library entries."""
        if not self._entries:
            if not self.load_index():
                self.scan()
        return self._entries

    def by_category(self, category: str) -> List[SoundEntry]:
        """Return entries matching the given category."""
        return [
            e for e in self.list_all()
            if e.category.lower() == category.lower()
        ]

    def search(self, query: str) -> List[SoundEntry]:
        """Simple text search across names and tags."""
        query_lower = query.lower()
        results = []
        for entry in self.list_all():
            if (query_lower in entry.name.lower()
                    or any(query_lower in t for t in entry.tags)):
                results.append(entry)
        return results

    def count(self) -> int:
        return len(self.list_all())

    # ── Display ───────────────────────────────────────────────────────

    def print_summary(self) -> None:
        """Print a summary of the library contents."""
        entries = self.list_all()

        if not entries:
            console.print("[yellow]Библиотека звуков пуста.[/]")
            console.print("Добавьте аудио файлы в подпапки sounds/:")
            for cat in sorted(self.KNOWN_CATEGORIES):
                console.print(f"  📁 sounds/{cat}/")
            return

        table = Table(title="Библиотека звуков")
        table.add_column("Категория", style="cyan")
        table.add_column("Количество", style="green", justify="right")

        by_cat: Dict[str, int] = {}
        for e in entries:
            by_cat[e.category] = by_cat.get(e.category, 0) + 1

        for cat in sorted(by_cat):
            table.add_row(cat, str(by_cat[cat]))

        table.add_row("[bold]Всего[/]", f"[bold]{len(entries)}[/]")
        console.print(table)

    # ── Management ────────────────────────────────────────────────────

    def add_sound(self, path: str | Path, category: str = "foley") -> SoundEntry:
        """Add a single sound file to the library."""
        import shutil

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        cat_dir = self.sounds_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        dest = cat_dir / path.name
        shutil.copy2(path, dest)

        entry = SoundEntry(
            path=dest,
            name=path.stem,
            category=category,
            tags=self._tags_from_name(path.stem),
        )

        try:
            import soundfile as sf
            info = sf.info(str(dest))
            entry.duration_sec = info.duration
        except Exception:
            pass

        self._entries.append(entry)
        self._save_index()
        return entry

    def generate_basic_sounds(self) -> int:
        """Generate a small set of basic synthetic sounds for testing.

        Creates simple sine waves, noise, and tones so the library
        is not completely empty on first run.

        Returns:
            Number of sounds generated.
        """
        import soundfile as sf

        sr = self.config.sample_rate
        generated = 0

        sounds_to_generate = [
            # ── Alerts & Tones (Alarms, Beeps, Rings) ─────────────────
            ("alerts", "alarm_beep_digital", lambda: _alarm_beep_digital(sr, 3.0)),
            ("alerts", "alarm_beep_fast", lambda: _alarm_beep_fast(sr, 2.0)),
            ("alerts", "alarm_beep_single", lambda: _alarm_beep_single(sr, 0.25)),
            ("alerts", "alarm_clock_bell", lambda: _alarm_clock_bell(sr, 2.5)),
            ("alerts", "digital_timer_beep", lambda: _digital_timer_beep(sr, 1.5)),
            ("alerts", "phone_ring_digital", lambda: _phone_ring_digital(sr, 3.0)),
            ("alerts", "microwave_chime", lambda: _microwave_chime(sr, 1.2)),
            ("alerts", "ui_beep_high", lambda: _ui_beep_high(sr, 0.3)),

            # ── Foley & Physical Transients ───────────────────────────
            ("foley", "door_knock_wood", lambda: _door_knock_wood(sr, 0.8)),
            ("foley", "switch_click", lambda: _switch_click(sr, 0.2)),
            ("foley", "clock_tick", lambda: _clock_tick(sr, 1.5)),
            ("foley", "footstep_floor", lambda: _footstep_floor(sr, 0.5)),
            ("foley", "keyboard_tap", lambda: _keyboard_tap(sr, 0.2)),
            ("foley", "glass_clink", lambda: _glass_clink(sr, 0.6)),

            # ── Ambience & Backgrounds ────────────────────────────────
            ("ambient", "room_tone", lambda: _room_tone(sr, 5.0)),
            ("ambient", "white_noise_soft", lambda: _white_noise(sr, 5.0, 0.30)),
            ("ambient", "pink_noise_soft", lambda: _pink_noise(sr, 5.0, 0.35)),
            ("ambient", "brown_noise", lambda: _brown_noise(sr, 5.0, 0.40)),
            ("nature", "wind_gentle", lambda: _wind_sound(sr, 5.0, 0.35)),
            ("nature", "rain_light", lambda: _rain_sound(sr, 5.0, 0.40)),
            ("urban", "city_hum", lambda: _city_hum(sr, 5.0)),
            ("urban", "traffic_distant", lambda: _traffic_distant(sr, 5.0)),
            ("music", "drone_low", lambda: _drone_low(sr, 5.0)),
            ("music", "pad_warm", lambda: _pad_warm(sr, 5.0)),
        ]

        for category, name, gen_func in sounds_to_generate:
            cat_dir = self.sounds_dir / category
            cat_dir.mkdir(parents=True, exist_ok=True)
            out_path = cat_dir / f"{name}.wav"
            try:
                audio = gen_func()
                sf.write(str(out_path), audio, sr)
                generated += 1
            except Exception as e:
                console.print(f"[dim red]  Не удалось сгенерировать {name}: {e}[/]")

        if generated > 0:
            self.scan()  # re-index

        return generated


# ── Synthetic sound generators ────────────────────────────────────────
# These produce simple audio arrays for the basic library.

def _white_noise(sr: int, duration: float, amplitude: float = 0.35) -> np.ndarray:
    samples = int(sr * duration)
    return (np.random.randn(samples).astype(np.float32) * amplitude)


def _pink_noise(sr: int, duration: float, amplitude: float = 0.40) -> np.ndarray:
    """Generate pink noise (1/f) using the Voss-McCartney algorithm."""
    samples = int(sr * duration)
    n_rows = 16
    array = np.empty((samples, n_rows))
    array.fill(np.nan)
    array[0, :] = np.random.randn(n_rows)
    steps = range(1, samples)
    cols = np.random.geometric(0.5, samples).clip(0, n_rows - 1)

    for i in steps:
        array[i] = array[i - 1]
        array[i, cols[i]] = np.random.randn()

    result = np.nansum(array, axis=1)
    result = result / (np.max(np.abs(result)) + 1e-8) * amplitude
    return result.astype(np.float32)


def _brown_noise(sr: int, duration: float, amplitude: float = 0.45) -> np.ndarray:
    samples = int(sr * duration)
    white = np.random.randn(samples)
    brown = np.cumsum(white)
    brown = brown / (np.max(np.abs(brown)) + 1e-8) * amplitude
    return brown.astype(np.float32)


def _room_tone(sr: int, duration: float) -> np.ndarray:
    """Low-level noise simulating room tone."""
    samples = int(sr * duration)
    noise = np.random.randn(samples)
    from scipy.signal import butter, lfilter
    b, a = butter(4, 800 / (sr / 2), btype="low")
    filtered = lfilter(b, a, noise)
    return (filtered / (np.max(np.abs(filtered)) + 1e-8) * 0.35).astype(np.float32)


def _wind_sound(sr: int, duration: float, amplitude: float = 0.45) -> np.ndarray:
    """Simulate gentle wind with filtered noise and amplitude modulation."""
    samples = int(sr * duration)
    noise = np.random.randn(samples)

    from scipy.signal import butter, lfilter
    b, a = butter(3, [100 / (sr / 2), 1500 / (sr / 2)], btype="band")
    filtered = lfilter(b, a, noise)

    t = np.linspace(0, duration, samples)
    mod = 0.5 + 0.5 * np.sin(2 * np.pi * 0.3 * t + np.random.uniform(0, 2 * np.pi))
    result = filtered * mod
    return (result / (np.max(np.abs(result)) + 1e-8) * amplitude).astype(np.float32)


def _rain_sound(sr: int, duration: float, amplitude: float = 0.50) -> np.ndarray:
    """Simulate light rain with filtered noise and random impacts."""
    samples = int(sr * duration)
    continuous = np.random.randn(samples) * 0.5

    from scipy.signal import butter, lfilter
    b, a = butter(2, [500 / (sr / 2), 5000 / (sr / 2)], btype="band")
    continuous = lfilter(b, a, continuous)

    n_drops = int(duration * 25)
    for _ in range(n_drops):
        pos = np.random.randint(0, max(1, samples - 500))
        drop_len = np.random.randint(50, 500)
        drop = np.random.randn(drop_len) * np.random.uniform(0.3, 0.8)
        drop *= np.exp(-np.linspace(0, 5, drop_len))
        end = min(pos + drop_len, samples)
        continuous[pos:end] += drop[: end - pos]

    return (continuous / (np.max(np.abs(continuous)) + 1e-8) * amplitude).astype(np.float32)


def _city_hum(sr: int, duration: float) -> np.ndarray:
    """City background hum."""
    samples = int(sr * duration)
    t = np.linspace(0, duration, samples)
    hum = (
        0.5 * np.sin(2 * np.pi * 60 * t)
        + 0.3 * np.sin(2 * np.pi * 120 * t)
        + 0.2 * np.sin(2 * np.pi * 180 * t)
        + np.random.randn(samples) * 0.05
    )
    return (hum / (np.max(np.abs(hum)) + 1e-8) * 0.40).astype(np.float32)


def _traffic_distant(sr: int, duration: float) -> np.ndarray:
    """Distant traffic rumble."""
    samples = int(sr * duration)
    noise = np.random.randn(samples)

    from scipy.signal import butter, lfilter
    b, a = butter(3, 400 / (sr / 2), btype="low")
    filtered = lfilter(b, a, noise)

    t = np.linspace(0, duration, samples)
    mod = 0.6 + 0.4 * np.sin(2 * np.pi * 0.15 * t)
    result = filtered * mod
    return (result / (np.max(np.abs(result)) + 1e-8) * 0.40).astype(np.float32)


def _room_ambience(sr: int, duration: float) -> np.ndarray:
    """Generic room ambience."""
    samples = int(sr * duration)
    noise = np.random.randn(samples)

    from scipy.signal import butter, lfilter
    b, a = butter(2, 2000 / (sr / 2), btype="low")
    filtered = lfilter(b, a, noise)
    return (filtered / (np.max(np.abs(filtered)) + 1e-8) * 0.35).astype(np.float32)


def _subtle_pad(sr: int, duration: float) -> np.ndarray:
    """A warm pad sound."""
    samples = int(sr * duration)
    t = np.linspace(0, duration, samples)
    pad = (
        0.4 * np.sin(2 * np.pi * 220 * t)
        + 0.3 * np.sin(2 * np.pi * 330 * t)
        + 0.2 * np.sin(2 * np.pi * 440 * t)
    )
    envelope = np.ones(samples)
    fade_len = int(sr * 0.5)
    envelope[:fade_len] = np.linspace(0, 1, fade_len)
    envelope[-fade_len:] = np.linspace(1, 0, fade_len)
    result = pad * envelope
    return (result / (np.max(np.abs(result)) + 1e-8) * 0.45).astype(np.float32)


def _drone_low(sr: int, duration: float) -> np.ndarray:
    """A low drone sound."""
    samples = int(sr * duration)
    t = np.linspace(0, duration, samples)
    drone = (
        0.5 * np.sin(2 * np.pi * 65 * t)
        + 0.3 * np.sin(2 * np.pi * 130 * t + 0.5)
        + 0.2 * np.sin(2 * np.pi * 195 * t + 1.0)
    )
    envelope = np.ones(samples)
    fade_len = int(sr * 0.8)
    envelope[:fade_len] = np.linspace(0, 1, fade_len)
    envelope[-fade_len:] = np.linspace(1, 0, fade_len)
    result = drone * envelope
    return (result / (np.max(np.abs(result)) + 1e-8) * 0.45).astype(np.float32)


def _pad_warm(sr: int, duration: float) -> np.ndarray:
    """A warm musical pad."""
    samples = int(sr * duration)
    t = np.linspace(0, duration, samples)
    pad = (
        0.35 * np.sin(2 * np.pi * 261.63 * t)
        + 0.30 * np.sin(2 * np.pi * 329.63 * t)
        + 0.25 * np.sin(2 * np.pi * 392.00 * t)
        + 0.15 * np.sin(2 * np.pi * 523.25 * t)
    )
    vibrato = 1.0 + 0.01 * np.sin(2 * np.pi * 4 * t)
    pad = pad * vibrato
    envelope = np.ones(samples)
    fade_len = int(sr * 1.0)
    envelope[:fade_len] = np.linspace(0, 1, fade_len)
    envelope[-fade_len:] = np.linspace(1, 0, fade_len)
    result = pad * envelope
    return (result / (np.max(np.abs(result)) + 1e-8) * 0.45).astype(np.float32)


# ── Alerts & Tonal Generators ─────────────────────────────────────────

def _alarm_beep_digital(sr: int, duration: float = 3.0) -> np.ndarray:
    """Classic digital watch/clock alarm beeping at 2800 Hz in pulses."""
    samples = int(sr * duration)
    out = np.zeros(samples, dtype=np.float32)
    beep_len = int(sr * 0.12)
    pause_len = int(sr * 0.08)
    group_pause = int(sr * 0.45)
    t = np.linspace(0, 0.12, beep_len, endpoint=False)
    single_beep = (0.75 * np.sin(2 * np.pi * 2800 * t) + 0.25 * np.sin(2 * np.pi * 5600 * t)) * np.hanning(beep_len)

    pos = 0
    beep_in_group = 0
    while pos + beep_len <= samples:
        out[pos : pos + beep_len] += single_beep
        beep_in_group += 1
        if beep_in_group % 4 == 0:
            pos += beep_len + group_pause
        else:
            pos += beep_len + pause_len
    return out


def _alarm_beep_fast(sr: int, duration: float = 2.0) -> np.ndarray:
    """Rapid electronic alarm beep at 2500 Hz."""
    samples = int(sr * duration)
    out = np.zeros(samples, dtype=np.float32)
    beep_len = int(sr * 0.09)
    pause_len = int(sr * 0.06)
    t = np.linspace(0, 0.09, beep_len, endpoint=False)
    single_beep = (0.8 * np.sin(2 * np.pi * 2500 * t) + 0.2 * np.sin(2 * np.pi * 5000 * t)) * np.hanning(beep_len)

    pos = 0
    while pos + beep_len <= samples:
        out[pos : pos + beep_len] += single_beep
        pos += beep_len + pause_len
    return out


def _alarm_beep_single(sr: int, duration: float = 0.25) -> np.ndarray:
    """A single clean electronic beep at 2600 Hz."""
    samples = int(sr * duration)
    t = np.linspace(0, duration, samples, endpoint=False)
    beep = (0.8 * np.sin(2 * np.pi * 2600 * t) + 0.2 * np.sin(2 * np.pi * 5200 * t)) * np.hanning(samples)
    return beep.astype(np.float32)


def _alarm_clock_bell(sr: int, duration: float = 2.5) -> np.ndarray:
    """Ringing twin-bell mechanical alarm clock (rapid hammer oscillation)."""
    samples = int(sr * duration)
    t = np.linspace(0, duration, samples, endpoint=False)
    ring_carrier = 0.6 * np.sin(2 * np.pi * 1850 * t) + 0.4 * np.sin(2 * np.pi * 2400 * t)
    hammer_mod = 0.5 + 0.5 * np.sin(2 * np.pi * 28 * t)  # 28 Hz strike rate
    decay = np.exp(-t * 0.4)
    out = ring_carrier * hammer_mod * decay
    return (out / (np.max(np.abs(out)) + 1e-8) * 0.75).astype(np.float32)


def _digital_timer_beep(sr: int, duration: float = 1.5) -> np.ndarray:
    """Double-beep kitchen or electronic timer tone at 2000 Hz."""
    samples = int(sr * duration)
    out = np.zeros(samples, dtype=np.float32)
    beep_len = int(sr * 0.15)
    t = np.linspace(0, 0.15, beep_len, endpoint=False)
    beep = np.sin(2 * np.pi * 2000 * t) * np.hanning(beep_len)

    for offset in [0.0, 0.25, 0.8, 1.05]:
        st = int(offset * sr)
        if st + beep_len <= samples:
            out[st : st + beep_len] += beep * 0.75
    return out


def _phone_ring_digital(sr: int, duration: float = 3.0) -> np.ndarray:
    """Dual-tone phone ring (440 Hz + 480 Hz) with standard cadence."""
    samples = int(sr * duration)
    t = np.linspace(0, duration, samples, endpoint=False)
    ring = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 480 * t)
    # Cadence: 1.5s on, 1.5s off
    gate = (np.sin(2 * np.pi * (1.0 / 3.0) * t) > 0).astype(np.float32)
    fade = np.hanning(int(sr * 0.05))
    out = ring * gate
    return (out * 0.70).astype(np.float32)


def _microwave_chime(sr: int, duration: float = 1.2) -> np.ndarray:
    """Clean appliance microwave / elevator chime at 1500 Hz with harmonic decay."""
    samples = int(sr * duration)
    t = np.linspace(0, duration, samples, endpoint=False)
    env = np.exp(-t * 4.5)
    chime = (0.6 * np.sin(2 * np.pi * 1500 * t) + 0.3 * np.sin(2 * np.pi * 3000 * t)) * env
    return (chime / (np.max(np.abs(chime)) + 1e-8) * 0.75).astype(np.float32)


def _ui_beep_high(sr: int, duration: float = 0.3) -> np.ndarray:
    """High subtle digital notification / click tone at 3200 Hz."""
    samples = int(sr * duration)
    t = np.linspace(0, duration, samples, endpoint=False)
    env = np.exp(-t * 22.0)
    out = np.sin(2 * np.pi * 3200 * t) * env
    return (out * 0.70).astype(np.float32)


# ── Foley & Physical Transients ───────────────────────────────────────

def _door_knock_wood(sr: int, duration: float = 0.8) -> np.ndarray:
    """Realistic wooden door double knock."""
    samples = int(sr * duration)
    out = np.zeros(samples, dtype=np.float32)
    for knock_time in [0.05, 0.28]:
        st = int(knock_time * sr)
        n = int(0.18 * sr)
        t = np.linspace(0, 0.18, n, endpoint=False)
        decay = np.exp(-t * 35.0)
        body = (0.5 * np.sin(2 * np.pi * 175 * t) + 0.3 * np.sin(2 * np.pi * 310 * t) + 0.2 * np.random.randn(n)) * decay
        if st + n <= samples:
            out[st : st + n] += body
    return (out / (np.max(np.abs(out)) + 1e-8) * 0.80).astype(np.float32)


def _switch_click(sr: int, duration: float = 0.2) -> np.ndarray:
    """Sharp mechanical light switch or relay click."""
    samples = int(sr * duration)
    t = np.linspace(0, duration, samples, endpoint=False)
    snap = np.exp(-t * 90.0) * (0.6 * np.sin(2 * np.pi * 3500 * t) + 0.4 * np.random.randn(samples))
    return (snap / (np.max(np.abs(snap)) + 1e-8) * 0.75).astype(np.float32)


def _clock_tick(sr: int, duration: float = 1.5) -> np.ndarray:
    """Mechanical clock ticks (tick-tock cadence)."""
    samples = int(sr * duration)
    out = np.zeros(samples, dtype=np.float32)
    for i, tick_time in enumerate([0.1, 0.85]):
        st = int(tick_time * sr)
        n = int(0.08 * sr)
        t = np.linspace(0, 0.08, n, endpoint=False)
        decay = np.exp(-t * 80.0)
        freq = 2400 if i % 2 == 0 else 1900
        tick = (0.7 * np.sin(2 * np.pi * freq * t) + 0.3 * np.random.randn(n)) * decay
        if st + n <= samples:
            out[st : st + n] += tick
    return (out / (np.max(np.abs(out)) + 1e-8) * 0.70).astype(np.float32)


def _footstep_floor(sr: int, duration: float = 0.5) -> np.ndarray:
    """Subtle wooden floor footstep."""
    samples = int(sr * duration)
    t = np.linspace(0, duration, samples, endpoint=False)
    decay = np.exp(-t * 22.0)
    low_thump = 0.7 * np.sin(2 * np.pi * 95 * t) * np.exp(-t * 18.0)
    creak = 0.3 * np.sin(2 * np.pi * 420 * t) * decay
    out = (low_thump + creak)
    return (out / (np.max(np.abs(out)) + 1e-8) * 0.65).astype(np.float32)


def _keyboard_tap(sr: int, duration: float = 0.2) -> np.ndarray:
    """Clean mechanical keyboard key tap."""
    samples = int(sr * duration)
    t = np.linspace(0, duration, samples, endpoint=False)
    click = (0.5 * np.sin(2 * np.pi * 4200 * t) + 0.5 * np.random.randn(samples)) * np.exp(-t * 70.0)
    return (click / (np.max(np.abs(click)) + 1e-8) * 0.70).astype(np.float32)


def _glass_clink(sr: int, duration: float = 0.6) -> np.ndarray:
    """High metallic/glass toast clink at 3100 Hz."""
    samples = int(sr * duration)
    t = np.linspace(0, duration, samples, endpoint=False)
    clink = (0.6 * np.sin(2 * np.pi * 3100 * t) + 0.4 * np.sin(2 * np.pi * 4650 * t)) * np.exp(-t * 9.0)
    return (clink / (np.max(np.abs(clink)) + 1e-8) * 0.70).astype(np.float32)
