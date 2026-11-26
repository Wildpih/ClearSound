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
        "Babbling", "Whispering", "Shout", "Yell", "Screaming", "Battle cry",
        "Singing", "Chant", "Male singing", "Female singing", "Child singing", "Synthetic singing",
        "Laughter", "Baby laughter", "Belly laugh", "Chuckle, chortle",
        "Crying, sobbing", "Baby cry, infant cry", "Whimper",
        "Breathing", "Gasp", "Pant", "Snort", "Sigh", "Yawn",
        "Groan", "Grunt", "Roar",
        "Cough", "Throat clearing", "Sneeze",
        "Chewing, mastication", "Biting", "Gulp", "Swallowing",
        "Burping, eructation", "Hiccup",
        "Chatter", "Hubbub, speech noise, speech babble",
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
        "Glass", "Shatter", "Whack, thwack", "Crack",
    }
    crowd = {
        "Crowd", "Cheering", "Applause", "Children playing",
    }
    camera = {
        "Camera", "Single-lens reflex camera", "Shutter", "Photographic film",
    }
    alert = {
        "Alarm", "Alarm clock", "Beep, bleep", "Buzzer",
        "Siren", "Telephone bell ringing", "Ringtone",
        "Smoke detector, smoke alarm", "Fire alarm", "Chime", "Bell",
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
        vocals: Optional[np.ndarray] = None,
        no_vocals: Optional[np.ndarray] = None,
        callback=None,
    ) -> Tuple[List[SoundEvent], List[Tuple[float, float]]]:
        """Analyze audio and produce a complete sound map.

        Args:
            audio: Original audio (channels, samples) or (samples,).
            sr: Sample rate.
            vocals: Demucs separated vocals stem (optional).
            no_vocals: Demucs separated non-vocal track (optional).
            callback: Optional status callback.

        Returns:
            (sound_events, speech_regions)
            - sound_events: List of all detected non-speech sound events.
            - speech_regions: List of (start_sec, end_sec) for all human vocal/body regions.
        """
        console.print("\n[bold cyan]🧠 Нейросетевой анализ звуковой карты (на ОРИГИНАЛЬНОМ аудио)...[/]")

        if audio.ndim > 1:
            audio_mono = np.mean(audio, axis=0)
        else:
            audio_mono = audio.copy()

        # ── Step 1: Detect speech & human sounds (Silero VAD + PANNs + Demucs Vocals) ──
        if callback:
            callback("Определяем границы речи и человеческих звуков...")
        silero_regions = self._detect_speech(audio_mono, sr)

        # ── Step 2: Detect all sound events with PANNs ─────────────────
        if callback:
            callback("Нейросеть PANNs анализирует все звуки...")
        raw_events = self._detect_sound_events(audio_mono, sr)

        # Extract PANNs events classified as speech/human sound (breathing, grunts, chewing, etc.)
        panns_speech_events = [e for e in raw_events if e.category == "speech"]
        panns_speech_regions = [(e.start_sec, e.end_sec) for e in panns_speech_events]

        # Extract vocal energy regions from Demucs vocals stem (captures breaths, mouth noises, whispers)
        vocal_energy_regions = []
        if vocals is not None:
            vocal_energy_regions = self._detect_vocals_energy(vocals, sr, threshold=0.005, pad_sec=0.10)

        # Merge all human sound sources to form the complete Human Vocal Mask
        all_human_regions = self._merge_regions(
            silero_regions + panns_speech_regions + vocal_energy_regions,
            merge_gap=0.18,
        )
        # Filter minimum duration 0.20s
        speech_regions = [r for r in all_human_regions if r[1] - r[0] >= 0.20]
        console.print(f"  Обнаружено сегментов с человеческими звуками: [bold]{len(speech_regions)}[/]")

        # ── Step 3: Non-speech events from neural network (PANNs) ──────
        raw_non_speech = [e for e in raw_events if e.category != "speech"]

        # If no_vocals stem is available, also run neural PANNs on it to detect
        # any real background sounds that were masked by loud speech
        if no_vocals is not None:
            nv_mono = np.mean(no_vocals, axis=0) if no_vocals.ndim > 1 else no_vocals
            nv_events = self._detect_sound_events(nv_mono, sr)
            for nev in nv_events:
                if nev.category != "speech" and nev.confidence >= 0.25:
                    already = any(
                        ev.category == nev.category and ev.overlaps_with(nev.start_sec - 0.3, nev.end_sec + 0.3)
                        for ev in raw_non_speech
                    )
                    if not already:
                        raw_non_speech.append(nev)

        non_speech_events = self._deduplicate_events(raw_non_speech)
        console.print(f"  Обнаружено не-речевых звуковых событий: [bold]{len(non_speech_events)}[/]")

        # ── Step 4: Mark events that overlap with speech/human sounds ───
        for event in non_speech_events:
            event.overlaps_speech = any(
                event.overlaps_with(sp_start, sp_end) for sp_start, sp_end in speech_regions
            )

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
        if audio_mono.ndim > 1:
            audio_mono = np.mean(audio_mono, axis=0)

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
                threshold = 0.45
                min_duration_sec = 0.50
            elif category == "crowd":
                threshold = 0.35
                min_duration_sec = 0.40
            elif category in ("gunshot", "explosion", "impact", "camera"):
                threshold = 0.25
                min_duration_sec = 0.08
            elif category in ("alert", "foley"):
                threshold = 0.25
                min_duration_sec = 0.08
            elif category == "other":
                threshold = 0.40  # very strict for unknown categories
                min_duration_sec = 0.25
            else:
                threshold = 0.30
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

    def _detect_vocals_energy(
        self,
        vocals: Optional[np.ndarray],
        sr: int,
        threshold: float = 0.005,
        pad_sec: float = 0.10,
    ) -> List[Tuple[float, float]]:
        """Detect human vocal energy from Demucs vocals stem (words, breaths, grunts, chewing, etc.)."""
        if vocals is None:
            return []
        v_mono = np.mean(vocals, axis=0) if vocals.ndim > 1 else vocals
        frame_len = int(0.020 * sr)
        hop_len = int(0.010 * sr)
        total_samples = len(v_mono)
        num_frames = (total_samples - frame_len) // hop_len
        if num_frames <= 0:
            return []

        from numpy.lib.stride_tricks import as_strided
        shape = (num_frames, frame_len)
        strides = (v_mono.strides[0] * hop_len, v_mono.strides[0])
        frames = as_strided(v_mono, shape=shape, strides=strides)
        rms = np.sqrt(np.mean(frames ** 2, axis=1))

        active = rms > threshold
        if not np.any(active):
            return []

        regions: List[Tuple[float, float]] = []
        in_r = False
        st = 0.0
        for i, a in enumerate(active):
            t = i * hop_len / sr
            if a and not in_r:
                st = max(0.0, t - pad_sec)
                in_r = True
            elif not a and in_r:
                en = min(total_samples / sr, t + pad_sec)
                regions.append((st, en))
                in_r = False
        if in_r:
            regions.append((st, total_samples / sr))

        return self._merge_regions(regions, merge_gap=0.15)

    def _detect_transient_events(self, audio: np.ndarray, sr: int) -> List[SoundEvent]:
        """Detect physical transient sounds (clicks, typing, gunshots, impacts, beeps)
        via onset and spectral feature analysis."""
        from clearsound.core.audio_utils import (
            compute_crest_factor, compute_spectral_centroid,
            compute_spectral_flatness, compute_tonal_signature,
        )
        audio_mono = np.mean(audio, axis=0) if audio.ndim > 1 else audio
        events = []
        onset_frames = librosa.onset.onset_detect(
            y=audio_mono, sr=sr, hop_length=512, backtrack=True, units='frames',
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)

        for onset_sec in onset_times:
            start = int(onset_sec * sr)
            end = min(start + int(0.25 * sr), len(audio_mono))
            seg = audio_mono[start:end]
            if len(seg) < 512:
                continue
            rms = float(np.sqrt(np.mean(seg ** 2)))
            if rms < 0.005:
                continue
            crest = compute_crest_factor(seg)
            centroid = float(np.mean(compute_spectral_centroid(seg, sr)))
            _, _, is_tonal = compute_tonal_signature(seg, sr)

            if is_tonal:
                label, cat = "Beep, bleep", "alert"
            elif crest > 8.0 and centroid > 2200:
                label, cat = "Gunshot, gunfire", "gunshot"
            elif crest > 7.0 and centroid < 600:
                label, cat = "Explosion", "explosion"
            elif centroid >= 2000 and centroid <= 5500 and crest > 4.5:
                label, cat = "Camera", "camera"
            elif crest > 4.5 and centroid < 2000:
                label, cat = "Thump, thud", "impact"
            elif crest > 3.8 and centroid > 3000:
                label, cat = "Typing", "foley"
            else:
                continue

            events.append(SoundEvent(
                label=label, category=cat,
                start_sec=onset_sec, end_sec=onset_sec + min(0.3, len(seg) / sr),
                confidence=min(0.95, rms * 6 + (crest / 20.0) * 0.3),
            ))
        return events

    def _spectral_fallback(self, audio_mono: np.ndarray, sr: int) -> List[SoundEvent]:
        """Fallback spectral analysis when PANNs is unavailable."""
        return self._detect_transient_events(audio_mono, sr)

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
