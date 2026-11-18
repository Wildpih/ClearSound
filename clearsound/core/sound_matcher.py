"""Sound matcher — searches the sound library for the best-fitting
replacement sounds to fill gaps left by vocal removal.

Uses spectral similarity (MFCC cosine distance) and category hints
to rank candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from sklearn.metrics.pairwise import cosine_similarity

from clearsound.config import Config
from clearsound.core.audio_utils import (
    compute_dominant_frequency,
    compute_mfcc,
    compute_spectral_envelope,
    load_audio,
)
from clearsound.core.donor_extractor import DonorSound
from clearsound.core.gap_analyzer import GapInfo
from clearsound.library.sound_library import SoundEntry, SoundLibrary

console = Console()


@dataclass
class MatchResult:
    """A ranked candidate sound for a specific gap."""

    sound: SoundEntry
    similarity_score: float          # 0-1, higher is better
    category_match: bool             # does the category hint match?
    spectral_distance: float         # lower is better
    recommended_volume_db: float     # suggested volume adjustment
    is_donor: bool = False           # was extracted from the audio track itself


class SoundMatcher:
    """Find sounds from donors and library that best match each gap's acoustic profile."""

    def __init__(self, config: Config, library: SoundLibrary):
        self.config = config
        self.library = library
        self._feature_cache: Dict[str, np.ndarray] = {}

    def find_matches(
        self,
        gaps: List[GapInfo],
        donors: Optional[List[DonorSound]] = None,
    ) -> Dict[int, List[MatchResult]]:
        """Find the best matching sounds for each gap, prioritizing clean track donors.

        Args:
            gaps: List of GapInfo from the analyzer.
            donors: Optional list of clean sound donors extracted from the track.

        Returns:
            Dictionary mapping gap index → sorted list of MatchResult (best first).
        """
        console.print("\n[bold cyan]🔎 Ищем подходящие звуки (доноры из ролика + библиотека)...[/]")

        all_sounds = list(self.library.list_all())
        donor_entries = [d.to_sound_entry() for d in (donors or [])]
        combined_sounds = donor_entries + all_sounds

        if not combined_sounds:
            console.print("[bold red]❌ Нет доступных звуков для восстановления![/]")
            return {i: [] for i in range(len(gaps))}

        console.print(f"  Кандидатов: [bold]{len(donor_entries)}[/] доноров из трека, [bold]{len(all_sounds)}[/] из библиотеки")

        # Cache features
        self._precompute_features(combined_sounds)

        # Match each gap
        results: Dict[int, List[MatchResult]] = {}

        for i, gap in enumerate(gaps):
            matches = self._match_gap(gap, combined_sounds, donors or [])
            results[i] = matches[:self.config.top_matches]

        matched_count = sum(1 for m in results.values() if m)
        console.print(f"[bold green]✓ Найдены подходящие звуки для {matched_count}/{len(gaps)} сегментов[/]")

        return results

    def _precompute_features(self, sounds: List[SoundEntry]) -> None:
        """Load and cache MFCC features for library sounds."""
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                "  Индексируем библиотеку...", total=len(sounds)
            )
            for sound in sounds:
                key = str(sound.path)
                if key not in self._feature_cache:
                    try:
                        audio, sr = load_audio(
                            sound.path, sr=self.config.sample_rate, mono=True
                        )
                        mfcc = compute_mfcc(
                            audio, sr, n_mfcc=self.config.n_mfcc
                        )
                        self._feature_cache[key] = np.mean(mfcc, axis=1)
                    except Exception as e:
                        console.print(
                            f"[dim red]  Пропускаем {sound.path.name}: {e}[/]"
                        )
                progress.advance(task)

    def _match_gap(
        self,
        gap: GapInfo,
        sounds: List[SoundEntry],
        donors: List[DonorSound],
    ) -> List[MatchResult]:
        """Score donors and library sounds against a single gap, strictly honoring acoustic type."""
        candidates: List[MatchResult] = []

        # Find donor object lookup
        donor_by_path = {str(d.path): d for d in donors}

        for sound in sounds:
            key = str(sound.path)
            if key not in self._feature_cache:
                continue

            lib_mfcc = self._feature_cache[key]
            gap_vec = gap.mfcc_profile.reshape(1, -1)
            lib_vec = lib_mfcc.reshape(1, -1)

            min_dim = min(gap_vec.shape[1], lib_vec.shape[1])
            base_sim = float(
                cosine_similarity(gap_vec[:, :min_dim], lib_vec[:, :min_dim])[0, 0]
            )
            sim = (base_sim + 1.0) / 2.0

            is_donor = key in donor_by_path
            donor_obj = donor_by_path.get(key)
            name_lower = sound.name.lower()
            cat_lower = sound.category.lower()

            # ── 1. Acoustic Type Rules & Penalties ────────────────────
            # If gap is standard speech dialogue:
            if gap.event_type == "speech_dialogue":
                # Never insert beeps, alerts, gunshots, or clicks into normal speech!
                if cat_lower in ["alerts", "impacts"] or (donor_obj and (donor_obj.is_tonal or donor_obj.is_transient)):
                    sim = 0.01

            # If gap is a camera drop impact:
            elif gap.event_type == "camera_drop":
                if any(w in name_lower for w in ["camera_drop", "drop_impact", "camera_table", "drop_table"]):
                    sim = 0.98
                elif cat_lower == "foley" and "drop" in name_lower:
                    sim = 0.92
                elif cat_lower in ["foley", "impacts"]:
                    sim = 0.60 + 0.20 * sim
                else:
                    sim = 0.02

            # If gap is camera shutter clicks:
            elif gap.event_type == "camera_clicks":
                if any(w in name_lower for w in ["camera_shutter", "shutter_burst", "shutter_single", "camera_clicks"]):
                    sim = 0.98
                elif cat_lower == "foley" and any(w in name_lower for w in ["click", "switch", "tap"]):
                    sim = 0.75 + 0.15 * sim
                else:
                    sim = 0.02

            # If gap is combat gunshots:
            elif gap.event_type == "gunshots_combat":
                if any(w in name_lower for w in ["combat_rifle", "gunshot", "shotgun", "pistol"]):
                    sim = 0.98
                elif cat_lower == "impacts":
                    sim = 0.75 + 0.20 * sim
                else:
                    sim = 0.02

            # If gap is a tonal beep or alarm:
            elif gap.is_tonal or gap.event_type == "tonal_beep":
                # If donor from track has matching frequency:
                if is_donor and donor_obj and donor_obj.is_tonal:
                    freq_ratio = min(gap.dominant_freq, donor_obj.dominant_freq) / max(gap.dominant_freq, donor_obj.dominant_freq, 1.0)
                    if freq_ratio > 0.85:
                        # Genuine match from the same audio track!
                        sim = 0.98 + 0.01 * freq_ratio
                    else:
                        # Different frequency pitch — not a match!
                        sim = 0.05
                elif cat_lower == "alerts":
                    # Library alert sound (alarm, beep, ring)
                    sim = 0.60 + 0.25 * sim
                else:
                    # NEVER insert non-alert sounds into an alarm beep!
                    sim = 0.02

            # If gap is an explosive blast or gunshot/impact:
            elif gap.event_type in ["explosive_blast", "transient_impact"] or gap.category_hint == "impacts":
                if is_donor and donor_obj and (donor_obj.is_transient or donor_obj.crest_factor > 3.5):
                    sim = 0.92
                elif cat_lower == "impacts":
                    sim = 0.75 + 0.20 * sim
                else:
                    # NEVER insert alarm beeps, drones, or ambient noise into an explosion!
                    sim = 0.02

            # If gap is a transient foley (knock, click, tap):
            elif gap.is_transient or gap.event_type == "transient_foley":
                if is_donor and donor_obj and donor_obj.is_transient:
                    sim = 0.90
                elif cat_lower == "foley":
                    sim = 0.65 + 0.25 * sim
                else:
                    sim = 0.02

            # If gap is short (< 0.8s):
            if gap.duration_sec < 0.8:
                if any(w in name_lower for w in ["wind", "rain", "white_noise", "pink_noise", "brown_noise"]):
                    sim = 0.02

            # Standard category bonus
            cat_match = (cat_lower == gap.category_hint.lower())
            if cat_match and sim > 0.1:
                sim = min(1.0, sim + 0.10)

            # Recommend volume
            rec_vol = self._recommend_volume(gap, sound)

            if sim >= 0.20:
                candidates.append(MatchResult(
                    sound=sound,
                    similarity_score=min(0.99, sim),
                    category_match=cat_match,
                    spectral_distance=1.0 - sim,
                    recommended_volume_db=rec_vol,
                    is_donor=is_donor,
                ))

        # Sort strictly by highest similarity score first
        candidates.sort(
            key=lambda m: -m.similarity_score
        )
        return candidates

    def _recommend_volume(self, gap: GapInfo, sound: SoundEntry) -> float:
        """Recommend a volume adjustment for the matched sound."""
        # Target: the energy loss should roughly be compensated
        if gap.energy_loss_ratio < 0.1:
            return -6.0  # minimal loss → keep quiet
        elif gap.energy_loss_ratio < 0.3:
            return -3.0
        elif gap.energy_loss_ratio < 0.6:
            return 0.0
        else:
            return min(3.0, self.config.max_boost_db)
