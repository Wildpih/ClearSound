"""Neural Sound Map Analyzer — Uses PANNs (Pretrained Audio Neural Networks)
to detect ALL sound events in the ORIGINAL audio BEFORE vocal removal.

Creates a comprehensive "sound map" — a timeline of every detected sound event
(speech, music, gunshots, impacts, crowd, etc.) with precise timestamps and
overlap information.

Key principle: Analyze the ORIGINAL audio so we know what sounds exist
BEFORE Demucs potentially degrades them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import torch
from rich.console import Console
from rich.table import Table

console = Console()

# ── AudioSet Label → Category Mapping ─────────────────────────────────

_CATEGORY_MAP: Dict[str, str] = {}

def _build_category_map() -> Dict[str, str]:
    """Build a mapping from AudioSet label → high-level category."""
    if _CATEGORY_MAP:
        return _CATEGORY_MAP

    speech = {
        "Speech", "Male speech, man speaking", "Female speech, woman speaking",
        "Child speech, kid speaking", "Conversation", "Narration, monologue",
        "Babbling", "Whispering", "Shout", "Yell", "Screaming",
        "Singing", "Chant", "Laughter", "Crying, sobbing",
    }
    music = {
        "Music", "Musical instrument", "Plucked string instrument",
        "Guitar", "Electric guitar", "Bass guitar", "Acoustic guitar",
        "Piano", "Keyboard (musical)", "Drum", "Drum kit", "Synthesizer",
        "Orchestra", "Choir", "Soundtrack", "Theme music", "Background music",
        "Pop music", "Rock music", "Hip hop music", "Electronic music",
        "Classical music", "Jazz", "Blues", "Reggae", "Country",
        "Techno", "Dubstep", "Disco", "Funk",
    }
    gunshot = {
        "Gunshot, gunfire", "Machine gun", "Fusillade", "Artillery fire", "Cap gun",
    }
    explosion = {
        "Explosion", "Burst, pop", "Bang", "Boom",
    }
    impact = {
        "Thump, thud", "Slam", "Knock", "Tap", "Hammer",
        "Crash", "Breaking", "Bouncing", "Smash, crash",
        "Glass", "Shatter",
    }
    crowd = {
        "Crowd", "Cheering", "Applause", "Chatter", "Hubbub, speech noise",
        "Children shouting",
    }
    camera = {
        "Camera", "Single-lens reflex camera", "Shutter",
    }
    alert = {
        "Alarm", "Alarm clock", "Beep, bleep", "Buzzer",
        "Siren", "Telephone bell ringing", "Ringtone",
        "Smoke detector, smoke alarm", "Fire alarm",
    }
    vehicle = {
        "Vehicle", "Car", "Engine", "Motor vehicle (road)",
        "Truck", "Bus", "Motorcycle", "Aircraft", "Helicopter",
    }
    nature = {
        "Wind", "Rain", "Thunder", "Thunderstorm", "Water",
        "Rain on surface", "Stream", "Ocean", "Waves, surf",
    }
    animal = {
        "Animal", "Dog", "Cat", "Bird", "Bird vocalization, bird call, bird song",
        "Chirp, tweet",
    }
    foley = {
        "Footsteps", "Door", "Creak", "Squeak", "Clatter",
        "Cutlery, silverware", "Dishes, pots, and pans",
        "Zipper (clothing)", "Keys jangling", "Typing",
        "Computer keyboard", "Writing",
    }

    for label_set, cat_name in [
        (speech, "speech"), (music, "music"), (gunshot, "gunshot"),
        (explosion, "explosion"), (impact, "impact"), (crowd, "crowd"),
        (camera, "camera"), (alert, "alert"), (vehicle, "vehicle"),
        (nature, "nature"), (animal, "animal"), (foley, "foley"),
    ]:
        for label in label_set:
            _CATEGORY_MAP[label] = cat_name

    return _CATEGORY_MAP


def classify_label(label: str) -> str:
    """Map an AudioSet label to a high-level category."""
    m = _build_category_map()
    return m.get(label, "other")


CATEGORY_ICONS = {
    "speech": "🗣️", "music": "🎵", "gunshot": "🔫", "explosion": "💥",
    "impact": "💢", "crowd": "👥", "camera": "📷", "alert": "⏰",
    "vehicle": "🚗", "nature": "🌧️", "animal": "🐾", "foley": "🎬",
    "other": "🔊",
}


# ── SoundEvent Data Class ─────────────────────────────────────────────

@dataclass
class SoundEvent:
    """A single detected sound event from the neural network."""

    label: str                 # AudioSet class name (e.g. "Gunshot, gunfire")
    category: str              # High-level category (e.g. "gunshot")
    start_sec: float           # Precise start time in seconds
    end_sec: float             # Precise end time in seconds
    confidence: float          # Detection confidence (0.0 - 1.0)
    overlaps_speech: bool = False   # Does this event overlap with speech?

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec

    @property
    def icon(self) -> str:
        return CATEGORY_ICONS.get(self.category, "🔊")

    def overlaps_with(self, start: float, end: float) -> bool:
        """Check if this event overlaps with the given time range."""
        return self.start_sec < end and self.end_sec > start


# ── Sound Map Analyzer ────────────────────────────────────────────────

class SoundMapAnalyzer:
    """Analyze original audio using neural networks to create a comprehensive
    sound map BEFORE vocal removal.

    Uses:
    1. PANNs Cnn14 for sound event detection (527 AudioSet classes)
    2. Silero VAD for precise speech boundary detection
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._sed_model = None
        self._vad_model = None
        self._vad_utils = None
        self._labels: Optional[List[str]] = None

    # ── Model Loading ─────────────────────────────────────────────────

    def _ensure_sed_model(self):
        """Load PANNs SoundEventDetection model."""
        if self._sed_model is not None:
            return self._sed_model

        console.print("[dim]  Загружаем PANNs Cnn14 для детекции звуковых событий...[/]")
        from panns_inference import SoundEventDetection
        self._sed_model = SoundEventDetection(
            checkpoint_path=None,  # auto-download
            device=self.device,
        )
        # Load labels
        from panns_inference.config import labels as panns_labels
        self._labels = panns_labels
        console.print("[green]  ✓ PANNs Cnn14 загружена[/]")
        return self._sed_model

    def _ensure_vad_model(self):
        """Load Silero VAD model for precise speech detection."""
        if self._vad_model is not None:
            return self._vad_model, self._vad_utils

        console.print("[dim]  Загружаем Silero VAD для детекции речи...[/]")
        try:
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False,
                trust_repo=True,
            )
            self._vad_model = model
            self._vad_utils = utils
            console.print("[green]  ✓ Silero VAD загружена[/]")
            return model, utils
        except Exception as e:
            console.print(f"[yellow]  ⚠ Silero VAD недоступна: {e}. Используем энергетический VAD.[/]")
            return None, None

    # ── Core Analysis ─────────────────────────────────────────────────

    def analyze(
        self,
        audio: np.ndarray,
        sr: int,
        callback=None,
    ) -> Tuple[List[SoundEvent], List[Tuple[float, float]]]:
        """Analyze original audio and produce a complete sound map.

        Args:
            audio: Original audio (channels, samples) or (samples,).
            sr: Sample rate.
            callback: Optional status callback.

        Returns:
            (sound_events, speech_regions)
            - sound_events: List of all detected non-speech sound events.
            - speech_regions: List of (start_sec, end_sec) for speech segments.
        """
        console.print("\n[bold cyan]🧠 Нейросетевой анализ звуковой карты (на ОРИГИНАЛЬНОМ аудио)...[/]")

        if audio.ndim > 1:
            audio_mono = np.mean(audio, axis=0)
        else:
            audio_mono = audio.copy()

        # ── Step 1: Detect speech with Silero VAD ─────────────────────
        if callback:
            callback("Определяем границы речи (Silero VAD)...")
        speech_regions = self._detect_speech(audio_mono, sr)
        console.print(f"  Обнаружено речевых сегментов: [bold]{len(speech_regions)}[/]")

        # ── Step 2: Detect all sound events with PANNs + Spectral Transient Analysis ─
        if callback:
            callback("Нейросеть PANNs анализирует все звуки...")
        raw_events = self._detect_sound_events(audio_mono, sr)

        # Merge in sharp physical transients from spectral analysis
        # to ensure no impacts/clicks/shots are missed even under loud speech
        spectral_events = self._spectral_fallback(audio_mono, sr)
        for sev in spectral_events:
            # Only add if not already covered by a PANNs event of similar category
            already_covered = any(
                ev.category == sev.category and ev.overlaps_with(sev.start_sec - 0.2, sev.end_sec + 0.2)
                for ev in raw_events
            )
            if not already_covered:
                raw_events.append(sev)

        raw_events = self._deduplicate_events(raw_events)
        console.print(f"  Обнаружено звуковых событий: [bold]{len(raw_events)}[/]")

        # ── Step 3: Mark events that overlap with speech ──────────────
        for event in raw_events:
            for sp_start, sp_end in speech_regions:
                if event.overlaps_with(sp_start, sp_end):
                    event.overlaps_speech = True
                    break

        # ── Step 4: Filter out pure speech events ─────────────────────
        non_speech_events = [e for e in raw_events if e.category != "speech"]

        self._print_summary(non_speech_events, speech_regions)
        return non_speech_events, speech_regions

    # ── Speech Detection ──────────────────────────────────────────────

    def _detect_speech(self, audio_mono: np.ndarray, sr: int) -> List[Tuple[float, float]]:
        vad_model, vad_utils = self._ensure_vad_model()
        if vad_model is not None:
            return self._silero_vad(audio_mono, sr, vad_model, vad_utils)
        return self._energy_vad(audio_mono, sr)

    def _silero_vad(self, audio_mono, sr, model, utils):
        get_speech_timestamps = utils[0]
        if sr != 16000:
            audio_16k = librosa.resample(audio_mono.astype(np.float32), orig_sr=sr, target_sr=16000)
        else:
            audio_16k = audio_mono.astype(np.float32)

        wav_tensor = torch.from_numpy(audio_16k)
        timestamps = get_speech_timestamps(
            wav_tensor, model, sampling_rate=16000,
            threshold=0.5, min_speech_duration_ms=250,
            min_silence_duration_ms=100, speech_pad_ms=30,
        )
        regions = [(ts["start"] / 16000.0, ts["end"] / 16000.0) for ts in timestamps]
        return self._merge_regions(regions, 0.15)

    def _energy_vad(self, audio_mono, sr):
        frame_len = int(0.025 * sr)
        hop_len = int(0.010 * sr)
        num_frames = max(0, (len(audio_mono) - frame_len) // hop_len)
        if num_frames == 0:
            return []
        energies = np.array([
            np.sqrt(np.mean(audio_mono[i * hop_len: i * hop_len + frame_len] ** 2))
            for i in range(num_frames)
        ])
        pos_e = energies[energies > 1e-6]
        threshold = np.percentile(pos_e, 30) if len(pos_e) > 0 else 0.01
        active = energies > threshold
        regions = []
        in_r = False
        r_start = 0.0
        for i, a in enumerate(active):
            t = i * hop_len / sr
            if a and not in_r:
                r_start = t
                in_r = True
            elif not a and in_r:
                regions.append((r_start, t))
                in_r = False
        if in_r:
            regions.append((r_start, len(audio_mono) / sr))
        return self._merge_regions(regions, 0.15)

    # ── Sound Event Detection ─────────────────────────────────────────

    def _detect_sound_events(self, audio_mono: np.ndarray, sr: int) -> List[SoundEvent]:
        """Detect sound events using PANNs SoundEventDetection on sliding windows."""
        try:
            sed = self._ensure_sed_model()
        except Exception as e:
            console.print(f"[yellow]  ⚠ PANNs недоступна: {e}. Используем спектральный анализ.[/]")
            return self._spectral_fallback(audio_mono, sr)

        labels = self._labels or []

        # PANNs expects 32kHz mono
        if sr != 32000:
            audio_32k = librosa.resample(audio_mono.astype(np.float32), orig_sr=sr, target_sr=32000)
        else:
            audio_32k = audio_mono.astype(np.float32)

        # Analyze in chunks (PANNs SED gives frame-level output)
        # Process the entire audio — SED handles windowing internally
        audio_tensor = torch.from_numpy(audio_32k[None, :]).float().to(self.device)

        with torch.no_grad():
            framewise_output = sed.inference(audio_tensor)
            # framewise_output: (1, time_frames, 527)

        if isinstance(framewise_output, np.ndarray):
            probs = framewise_output[0]  # (time_frames, 527)
        else:
            probs = framewise_output[0].cpu().numpy()
        num_frames = probs.shape[0]

        # Time resolution: PANNs uses hop_size=320 at 32kHz → 10ms per frame
        frame_duration = 320 / 32000.0  # 0.01 seconds

        # ── Extract events from probability matrix ────────────────────
        events: List[SoundEvent] = []

        for class_idx in range(len(labels)):
            label = labels[class_idx]
            category = classify_label(label)

            # Adaptive threshold & min duration by category
            if category == "speech":
                threshold = 0.40
                min_duration_sec = 0.25
            elif category == "music":
                threshold = 0.42
                min_duration_sec = 0.50
            elif category == "crowd":
                threshold = 0.35
                min_duration_sec = 0.40
            elif category in ("gunshot", "explosion", "impact", "camera"):
                threshold = 0.08
                min_duration_sec = 0.04
            elif category in ("alert", "foley"):
                threshold = 0.12
                min_duration_sec = 0.05
            elif category == "other":
                threshold = 0.50  # very strict for unknown categories
                min_duration_sec = 0.25
            else:
                threshold = 0.20
                min_duration_sec = 0.15

            class_probs = probs[:, class_idx]
            active = class_probs > threshold

            if not np.any(active):
                continue

            # Group consecutive active frames into events
            in_event = False
            ev_start = 0
            max_conf = 0.0

            for frame_idx in range(num_frames):
                if active[frame_idx] and not in_event:
                    ev_start = frame_idx
                    max_conf = float(class_probs[frame_idx])
                    in_event = True
                elif active[frame_idx] and in_event:
                    max_conf = max(max_conf, float(class_probs[frame_idx]))
                elif not active[frame_idx] and in_event:
                    start_sec = ev_start * frame_duration
                    end_sec = frame_idx * frame_duration
                    if end_sec - start_sec >= min_duration_sec:
                        events.append(SoundEvent(
                            label=label, category=category,
                            start_sec=start_sec, end_sec=end_sec,
                            confidence=max_conf,
                        ))
                    in_event = False

            if in_event:
                start_sec = ev_start * frame_duration
                end_sec = num_frames * frame_duration
                if end_sec - start_sec >= min_duration_sec:
                    events.append(SoundEvent(
                        label=label, category=category,
                        start_sec=start_sec, end_sec=end_sec,
                        confidence=max_conf,
                    ))

        events.sort(key=lambda e: (e.start_sec, -e.confidence))
        events = self._deduplicate_events(events)
        return events

    def _spectral_fallback(self, audio_mono: np.ndarray, sr: int) -> List[SoundEvent]:
        """Fallback spectral analysis when PANNs is unavailable."""
        from clearsound.core.audio_utils import (
            compute_crest_factor, compute_spectral_centroid,
            compute_spectral_flatness, compute_tonal_signature,
        )
        events = []
        onset_frames = librosa.onset.onset_detect(
            y=audio_mono, sr=sr, hop_length=512, backtrack=True, units='frames',
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)

        for onset_sec in onset_times:
            start = int(onset_sec * sr)
            end = min(start + int(0.5 * sr), len(audio_mono))
            seg = audio_mono[start:end]
            if len(seg) < 512:
                continue
            rms = float(np.sqrt(np.mean(seg ** 2)))
            if rms < 0.01:
                continue
            crest = compute_crest_factor(seg)
            centroid = float(np.mean(compute_spectral_centroid(seg, sr)))
            _, _, is_tonal = compute_tonal_signature(seg, sr)

            if is_tonal:
                label, cat = "Beep, bleep", "alert"
            elif crest > 10 and centroid > 2000:
                label, cat = "Gunshot, gunfire", "gunshot"
            elif crest > 8 and centroid < 500:
                label, cat = "Explosion", "explosion"
            elif crest > 6:
                label, cat = "Thump, thud", "impact"
            else:
                continue

            events.append(SoundEvent(
                label=label, category=cat,
                start_sec=onset_sec, end_sec=onset_sec + len(seg) / sr,
                confidence=min(0.9, rms * 5),
            ))
        return events

    # ── Utility Methods ───────────────────────────────────────────────

    @staticmethod
    def _merge_regions(regions, merge_gap=0.15):
        if not regions:
            return []
        s = sorted(regions, key=lambda r: r[0])
        merged = [s[0]]
        for start, end in s[1:]:
            ps, pe = merged[-1]
            if start - pe < merge_gap:
                merged[-1] = (ps, max(pe, end))
            else:
                merged.append((start, end))
        return merged

    @staticmethod
    def _deduplicate_events(events: List[SoundEvent]) -> List[SoundEvent]:
        if not events:
            return []
        by_cat: Dict[str, List[SoundEvent]] = {}
        for e in events:
            by_cat.setdefault(e.category, []).append(e)

        result = []
        for cat, cat_events in by_cat.items():
            cat_events.sort(key=lambda e: e.start_sec)
            merged = [cat_events[0]]
            for ev in cat_events[1:]:
                prev = merged[-1]
                if ev.start_sec < prev.end_sec + 0.5:
                    merged[-1] = SoundEvent(
                        label=prev.label if prev.confidence >= ev.confidence else ev.label,
                        category=cat,
                        start_sec=min(prev.start_sec, ev.start_sec),
                        end_sec=max(prev.end_sec, ev.end_sec),
                        confidence=max(prev.confidence, ev.confidence),
                        overlaps_speech=prev.overlaps_speech or ev.overlaps_speech,
                    )
                else:
                    merged.append(ev)
            result.extend(merged)
        result.sort(key=lambda e: (e.start_sec, -e.confidence))
        return result

    def _print_summary(self, events, speech_regions):
        from clearsound.core.audio_utils import format_time
        total_speech = sum(e - s for s, e in speech_regions)
        console.print(f"  Суммарная длительность речи: [bold]{total_speech:.1f}с[/] в {len(speech_regions)} сегментах")

        if events:
            table = Table(title="Нейросетевая карта звуков (не-речевые события)")
            table.add_column("#", style="dim", width=4)
            table.add_column("Время", style="cyan")
            table.add_column("Категория", style="bold")
            table.add_column("Звук", style="green")
            table.add_column("Увер.", style="yellow", width=6)
            table.add_column("+ Речь?", style="red", width=10)

            for i, ev in enumerate(events, 1):
                table.add_row(
                    str(i),
                    f"{format_time(ev.start_sec)} — {format_time(ev.end_sec)} ({ev.duration_sec:.1f}с)",
                    f"{ev.icon} {ev.category}",
                    ev.label,
                    f"{ev.confidence:.0%}",
                    "✗ Да" if ev.overlaps_speech else "✓ Нет",
                )
            console.print(table)

            cats = {}
            for e in events:
                cats[e.category] = cats.get(e.category, 0) + 1
            cat_str = ", ".join(f"{CATEGORY_ICONS.get(c, '🔊')} {c}: {n}" for c, n in sorted(cats.items()))
            console.print(f"  [dim]Категории: {cat_str}[/]")
            overlap_count = sum(1 for e in events if e.overlaps_speech)
            console.print(f"  Событий с речью: [bold]{overlap_count}/{len(events)}[/]")
