"""Sound matcher — searches the sound library for replacement sounds
using neural network labels and spectral similarity.

No hardcoded sound names or timestamp-based matching.
Uses AudioSet labels from the neural sound map to find appropriate
library sounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn
from sklearn.metrics.pairwise import cosine_similarity

from clearsound.config import Config
from clearsound.core.audio_utils import compute_mfcc, load_audio
from clearsound.core.donor_extractor import DonorSound
from clearsound.core.gap_analyzer import GapInfo
from clearsound.core.sound_map_analyzer import classify_label, CATEGORY_ICONS
from clearsound.library.sound_library import SoundEntry, SoundLibrary

console = Console()


@dataclass
class MatchResult:
    """A ranked candidate sound for a specific gap."""
    sound: SoundEntry
    similarity_score: float
    category_match: bool
    spectral_distance: float
    recommended_volume_db: float
    is_donor: bool = False


# Maps neural categories to keywords to search in library sound names.
CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "gunshot":    ["gunshot", "gun", "rifle", "pistol", "shotgun", "fire", "shoot"],
    "explosion":  ["explosion", "blast", "boom", "detonate", "bang"],
    "impact":     ["impact", "thump", "slam", "knock", "crash", "break", "hit", "punch", "smash"],
    "camera":     ["camera", "shutter", "click", "photo", "snap"],
    "alert":      ["alarm", "beep", "buzzer", "ring", "alert", "siren", "chime"],
    "crowd":      ["crowd", "cheer", "applause", "chatter"],
    "vehicle":    ["car", "engine", "motor", "truck", "vehicle", "drive"],
    "nature":     ["wind", "rain", "thunder", "water", "ocean", "storm"],
    "animal":     ["dog", "cat", "bird", "animal", "bark", "meow"],
    "foley":      ["footstep", "door", "creak", "step", "walk", "key", "type"],
    "music":      ["music", "melody", "instrument", "piano", "guitar", "drum", "orchestra"],
}


class SoundMatcher:
    """Find sounds from library that match each gap's neural classification."""

    def __init__(self, config: Config, library: SoundLibrary):
        self.config = config
        self.library = library
        self._feature_cache: Dict[str, np.ndarray] = {}

    def find_matches(
        self,
        gaps: List[GapInfo],
        donors: Optional[List[DonorSound]] = None,
    ) -> Dict[int, List[MatchResult]]:
        console.print("\n[bold cyan]🔎 Подбираем звуки по нейросетевым меткам...[/]")

        all_sounds = list(self.library.list_all())
        donor_entries = [d.to_sound_entry() for d in (donors or [])]
        combined = donor_entries + all_sounds

        if not combined:
            console.print("[bold red]❌ Нет доступных звуков![/]")
            return {i: [] for i in range(len(gaps))}

        console.print(f"  Кандидатов: [bold]{len(donor_entries)}[/] доноров, [bold]{len(all_sounds)}[/] из библиотеки")
        self._precompute_features(combined)

        results: Dict[int, List[MatchResult]] = {}
        for i, gap in enumerate(gaps):
            matches = self._match_gap(gap, combined, donors or [])
            results[i] = matches[:self.config.top_matches]

        matched = sum(1 for m in results.values() if m)
        console.print(f"[bold green]✓ Подобраны звуки для {matched}/{len(gaps)} сегментов[/]")
        return results

    def _precompute_features(self, sounds):
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(), console=console,
        ) as progress:
            task = progress.add_task("  Индексируем библиотеку...", total=len(sounds))
            for sound in sounds:
                key = str(sound.path)
                if key not in self._feature_cache:
                    try:
                        audio, sr = load_audio(sound.path, sr=self.config.sample_rate, mono=True)
                        mfcc = compute_mfcc(audio, sr, n_mfcc=self.config.n_mfcc)
                        self._feature_cache[key] = np.mean(mfcc, axis=1)
                    except Exception:
                        pass
                progress.advance(task)

    def _match_gap(self, gap, sounds, donors):
        candidates: List[MatchResult] = []

        # Pure speech or speech with only background music -> no SFX match needed
        # (Background music is handled continuously by music_track / no_vocals)
        sfx_events = [e for e in gap.overlapping_events if e.category not in ("music", "speech")]
        if not sfx_events:
            return []

        # Target the highest-confidence discrete SFX event
        target_sfx = max(sfx_events, key=lambda e: e.confidence)

        # Require genuine neural confidence (>= 0.25) for inserting discrete SFX
        if target_sfx.confidence < 0.25:
            return []

        target_cat = target_sfx.category
        target_kw = CATEGORY_KEYWORDS.get(target_cat, [])
        donor_paths = {str(d.path) for d in donors}

        for sound in sounds:
            cat_lower = sound.category.lower()
            name_lower = sound.name.lower()

            # Ambient noise / tone files must NEVER be inserted as discrete SFX!
            if cat_lower in ("ambient", "noise") or any(
                w in name_lower for w in ["white_noise", "pink_noise", "brown_noise", "room_tone", "drone", "pad"]
            ):
                continue

            # Never insert background music tracks as discrete SFX
            if cat_lower == "music" and target_cat != "music":
                continue

            key = str(sound.path)
            if key not in self._feature_cache:
                continue

            is_donor = key in donor_paths

            # Category/keyword match
            kw_match = any(kw in name_lower for kw in target_kw)
            cat_match = (cat_lower == target_cat)

            if kw_match:
                base = 0.80
            elif cat_match:
                base = 0.65
            elif is_donor:
                base = 0.50
            else:
                base = 0.10

            # Spectral similarity bonus
            lib_mfcc = self._feature_cache[key]
            gap_vec = gap.mfcc_profile.reshape(1, -1)
            lib_vec = lib_mfcc.reshape(1, -1)
            min_dim = min(gap_vec.shape[1], lib_vec.shape[1])
            cos_sim = float(cosine_similarity(gap_vec[:, :min_dim], lib_vec[:, :min_dim])[0, 0])
            bonus = max(0.0, (cos_sim + 1.0) / 2.0 - 0.5) * 0.2

            sim = min(0.99, base + bonus)

            # Penalty for clearly wrong categories
            if target_cat in ("gunshot", "explosion", "impact"):
                if any(w in name_lower for w in ["wind", "rain", "noise", "room_tone", "pad", "drone"]):
                    sim = 0.01
            if target_cat == "alert":
                if any(w in name_lower for w in ["explosion", "gunshot", "crash", "wind"]):
                    sim = 0.01

            rec_vol = self._recommend_volume(gap)

            if sim >= 0.50:
                candidates.append(MatchResult(
                    sound=sound, similarity_score=sim,
                    category_match=cat_match or kw_match,
                    spectral_distance=1.0 - sim,
                    recommended_volume_db=rec_vol, is_donor=is_donor,
                ))

        candidates.sort(key=lambda m: -m.similarity_score)
        return candidates

    def _recommend_volume(self, gap):
        if gap.energy_loss_ratio < 0.1:
            return -6.0
        elif gap.energy_loss_ratio < 0.3:
            return -3.0
        elif gap.energy_loss_ratio < 0.6:
            return 0.0
        else:
            return min(3.0, self.config.max_boost_db)
