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
    compute_mfcc,
    compute_spectral_envelope,
    load_audio,
)
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


class SoundMatcher:
    """Find sounds from the library that best match each gap's spectral
    profile."""

    def __init__(self, config: Config, library: SoundLibrary):
        self.config = config
        self.library = library
        self._feature_cache: Dict[str, np.ndarray] = {}

    def find_matches(self, gaps: List[GapInfo]
                     ) -> Dict[int, List[MatchResult]]:
        """Find the best matching sounds for each gap.

        Args:
            gaps: List of GapInfo from the analyzer.

        Returns:
            Dictionary mapping gap index → sorted list of MatchResult
            (best first).
        """
        console.print("\n[bold cyan]🔎 Ищем подходящие звуки в библиотеке...[/]")

        # ── Pre-compute features for all library sounds ───────────────
        all_sounds = self.library.list_all()
        if not all_sounds:
            console.print("[bold red]❌ Библиотека звуков пуста![/]")
            console.print("  Добавьте .wav / .mp3 / .flac файлы в папку sounds/")
            console.print("  Категории: ambient, music, nature, urban, foley")
            return {i: [] for i in range(len(gaps))}

        console.print(f"  Звуков в библиотеке: [bold]{len(all_sounds)}[/]")

        self._precompute_features(all_sounds)

        # ── Match each gap ────────────────────────────────────────────
        results: Dict[int, List[MatchResult]] = {}

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("  Подбираем звуки...", total=len(gaps))

            for i, gap in enumerate(gaps):
                matches = self._match_gap(gap, all_sounds)
                results[i] = matches[:self.config.top_matches]
                progress.advance(task)

        # Summary
        matched_count = sum(1 for m in results.values() if m)
        console.print(
            f"[bold green]✓ Найдены совпадения для "
            f"{matched_count}/{len(gaps)} сегментов[/]"
        )

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
        self, gap: GapInfo, sounds: List[SoundEntry]
    ) -> List[MatchResult]:
        """Score every library sound against a single gap and return
        sorted matches."""
        candidates: List[MatchResult] = []

        for sound in sounds:
            key = str(sound.path)
            if key not in self._feature_cache:
                continue

            lib_mfcc = self._feature_cache[key]

            # ── Cosine similarity on MFCC vectors ─────────────────────
            gap_vec = gap.mfcc_profile.reshape(1, -1)
            lib_vec = lib_mfcc.reshape(1, -1)

            # Ensure same dimensionality
            min_dim = min(gap_vec.shape[1], lib_vec.shape[1])
            sim = float(
                cosine_similarity(gap_vec[:, :min_dim], lib_vec[:, :min_dim])[0, 0]
            )
            # Normalize to 0-1 (cosine sim is already -1 to 1)
            sim = (sim + 1.0) / 2.0

            # ── Category bonus ────────────────────────────────────────
            cat_match = (
                sound.category.lower() == gap.category_hint.lower()
                if gap.category_hint and sound.category
                else False
            )
            if cat_match:
                sim = min(1.0, sim + 0.15)

            # ── Spectral distance (secondary) ─────────────────────────
            try:
                audio_lib, sr = load_audio(
                    sound.path, sr=self.config.sample_rate, mono=True
                )
                _, lib_env = compute_spectral_envelope(audio_lib, sr)
                if gap.spectral_envelope is not None:
                    min_len = min(len(gap.spectral_envelope), len(lib_env))
                    dist = float(np.mean(np.abs(
                        gap.spectral_envelope[:min_len] - lib_env[:min_len]
                    )))
                else:
                    dist = 1.0
            except Exception:
                dist = 1.0

            # ── Recommended volume ────────────────────────────────────
            rec_vol = self._recommend_volume(gap, sound)

            if sim >= self.config.min_similarity:
                candidates.append(MatchResult(
                    sound=sound,
                    similarity_score=sim,
                    category_match=cat_match,
                    spectral_distance=dist,
                    recommended_volume_db=rec_vol,
                ))

        # Sort: similarity desc, then spectral distance asc
        candidates.sort(
            key=lambda m: (-m.similarity_score, m.spectral_distance)
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
