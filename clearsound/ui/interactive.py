"""Interactive preview UI.

Lets the user listen to the processed audio, compare segments,
and provide feedback for iterative adjustment.
"""

from __future__ import annotations

import io
import struct
import tempfile
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
from rich.table import Table

from clearsound.config import Config
from clearsound.core.audio_utils import (
    format_time,
    save_audio,
    seconds_to_samples,
    samples_to_seconds,
)
from clearsound.core.gap_analyzer import GapInfo
from clearsound.core.sound_matcher import MatchResult

console = Console()


def _play_audio_array(audio: np.ndarray, sr: int) -> None:
    """Play a numpy audio array through the speakers.

    Uses simpleaudio if available, otherwise falls back to writing
    a temp WAV and using the system player.
    """
    # Ensure mono or stereo
    if audio.ndim > 1:
        play_data = np.mean(audio, axis=0)
    else:
        play_data = audio

    # Normalize to int16 range
    play_data = np.clip(play_data, -1.0, 1.0)
    int_data = (play_data * 32767).astype(np.int16)

    try:
        import simpleaudio as sa
        play_obj = sa.play_buffer(
            int_data.tobytes(), num_channels=1,
            bytes_per_sample=2, sample_rate=sr,
        )
        play_obj.wait_done()
    except ImportError:
        # Fallback: save to temp file and use system player
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        save_audio(play_data, tmp_path, sr)
        import subprocess
        import sys
        if sys.platform == "win32":
            subprocess.run(["powershell", "-c",
                            f"(New-Object Media.SoundPlayer '{tmp_path}').PlaySync()"],
                           capture_output=True)
        elif sys.platform == "darwin":
            subprocess.run(["afplay", tmp_path], capture_output=True)
        else:
            subprocess.run(["aplay", tmp_path], capture_output=True)


class InteractivePreview:
    """Interactive session for previewing and adjusting the result."""

    def __init__(self, config: Config):
        self.config = config

    def run(
        self,
        original: np.ndarray,
        no_vocals: np.ndarray,
        enhanced: np.ndarray,
        gaps: List[GapInfo],
        matches: Dict[int, List[MatchResult]],
        sr: int,
    ) -> Tuple[np.ndarray, Dict[int, float]]:
        """Run the interactive preview loop.

        Args:
            original: Original audio.
            no_vocals: Audio without vocals (before enhancement).
            enhanced: Enhanced audio (after sound insertion).
            gaps: List of gaps.
            matches: Match results per gap.
            sr: Sample rate.

        Returns:
            (final_audio, volume_adjustments) — the corrected audio
            and any volume overrides the user requested.
        """
        volume_adjustments: Dict[int, float] = {}
        current_audio = enhanced.copy()

        console.print(Panel(
            "[bold]Интерактивный просмотр результата[/]\n\n"
            "Команды:\n"
            "  [cyan]play[/]           — прослушать весь результат\n"
            "  [cyan]compare N[/]      — сравнить сегмент N (оригинал vs результат)\n"
            "  [cyan]louder N[/]       — сделать вставку в сегменте N громче\n"
            "  [cyan]quieter N[/]      — сделать вставку в сегменте N тише\n"
            "  [cyan]volume N X[/]     — установить громкость сегмента N на X дБ\n"
            "  [cyan]skip N[/]         — убрать вставку из сегмента N\n"
            "  [cyan]restore N[/]      — вернуть вставку в сегменте N\n"
            "  [cyan]list[/]           — показать все сегменты\n"
            "  [cyan]play_segment N[/] — прослушать сегмент N\n"
            "  [cyan]accept[/]         — принять результат и сохранить\n"
            "  [cyan]quit[/]           — выйти без сохранения\n",
            title="🎧 Предпрослушивание",
        ))

        skipped: set = set()
        iteration = 0

        while iteration < self.config.max_feedback_iterations:
            try:
                cmd = Prompt.ask("\n[bold cyan]>>[/]").strip().lower()
            except (EOFError, KeyboardInterrupt):
                break

            if not cmd:
                continue

            parts = cmd.split()
            action = parts[0]

            if action == "accept":
                console.print("[bold green]✓ Результат принят![/]")
                return current_audio, volume_adjustments

            elif action == "quit":
                console.print("[yellow]Выход без сохранения.[/]")
                return enhanced, volume_adjustments

            elif action == "play":
                console.print("[dim]Воспроизведение полного результата...[/]")
                _play_audio_array(current_audio, sr)

            elif action == "list":
                self._print_gaps(gaps, matches, volume_adjustments, skipped)

            elif action in ("compare", "play_segment") and len(parts) >= 2:
                try:
                    idx = int(parts[1]) - 1  # 1-based in UI
                    if 0 <= idx < len(gaps):
                        gap = gaps[idx]
                        ctx = seconds_to_samples(self.config.preview_context, sr)
                        start = max(0, gap.start_sample - ctx)
                        end = min(
                            self._get_length(current_audio),
                            gap.end_sample + ctx,
                        )

                        if action == "compare":
                            console.print(f"[dim]Оригинал (сегмент {idx+1}):[/]")
                            _play_audio_array(
                                self._get_segment(original, start, end), sr
                            )
                            console.print(f"[dim]Результат (сегмент {idx+1}):[/]")
                            _play_audio_array(
                                self._get_segment(current_audio, start, end), sr
                            )
                        else:
                            console.print(f"[dim]Сегмент {idx+1}:[/]")
                            _play_audio_array(
                                self._get_segment(current_audio, start, end), sr
                            )
                    else:
                        console.print(f"[red]Нет сегмента {idx+1}[/]")
                except ValueError:
                    console.print("[red]Укажите номер сегмента[/]")

            elif action == "louder" and len(parts) >= 2:
                try:
                    idx = int(parts[1]) - 1
                    current_adj = volume_adjustments.get(idx, 0.0)
                    volume_adjustments[idx] = current_adj + 3.0
                    console.print(
                        f"Сегмент {idx+1}: громкость +3 дБ "
                        f"(текущая: {volume_adjustments[idx]:+.1f} дБ)"
                    )
                    iteration += 1
                except ValueError:
                    console.print("[red]Укажите номер сегмента[/]")

            elif action == "quieter" and len(parts) >= 2:
                try:
                    idx = int(parts[1]) - 1
                    current_adj = volume_adjustments.get(idx, 0.0)
                    volume_adjustments[idx] = current_adj - 3.0
                    console.print(
                        f"Сегмент {idx+1}: громкость -3 дБ "
                        f"(текущая: {volume_adjustments[idx]:+.1f} дБ)"
                    )
                    iteration += 1
                except ValueError:
                    console.print("[red]Укажите номер сегмента[/]")

            elif action == "volume" and len(parts) >= 3:
                try:
                    idx = int(parts[1]) - 1
                    vol = float(parts[2])
                    volume_adjustments[idx] = vol
                    console.print(
                        f"Сегмент {idx+1}: громкость = {vol:+.1f} дБ"
                    )
                    iteration += 1
                except ValueError:
                    console.print("[red]Формат: volume <номер> <дБ>[/]")

            elif action == "skip" and len(parts) >= 2:
                try:
                    idx = int(parts[1]) - 1
                    skipped.add(idx)
                    volume_adjustments[idx] = -100.0  # effectively mute
                    console.print(f"Сегмент {idx+1}: вставка отключена")
                    iteration += 1
                except ValueError:
                    console.print("[red]Укажите номер сегмента[/]")

            elif action == "restore" and len(parts) >= 2:
                try:
                    idx = int(parts[1]) - 1
                    skipped.discard(idx)
                    volume_adjustments.pop(idx, None)
                    console.print(f"Сегмент {idx+1}: вставка восстановлена")
                    iteration += 1
                except ValueError:
                    console.print("[red]Укажите номер сегмента[/]")

            else:
                console.print(
                    "[dim]Неизвестная команда. Доступные: play, compare, "
                    "louder, quieter, volume, skip, restore, list, "
                    "play_segment, accept, quit[/]"
                )

        console.print("[yellow]Достигнут лимит итераций.[/]")
        return current_audio, volume_adjustments

    def _get_segment(self, audio: np.ndarray, start: int, end: int) -> np.ndarray:
        if audio.ndim > 1:
            return audio[:, start:end]
        return audio[start:end]

    def _get_length(self, audio: np.ndarray) -> int:
        if audio.ndim > 1:
            return audio.shape[1]
        return len(audio)

    def _print_gaps(
        self,
        gaps: List[GapInfo],
        matches: Dict[int, List[MatchResult]],
        volume_adjustments: Dict[int, float],
        skipped: set,
    ) -> None:
        table = Table(title="Сегменты")
        table.add_column("#", style="dim", width=4)
        table.add_column("Время", style="cyan")
        table.add_column("Длительность", style="green")
        table.add_column("Звук", style="magenta")
        table.add_column("Громкость", style="yellow")
        table.add_column("Статус", style="bold")

        for i, gap in enumerate(gaps):
            gap_matches = matches.get(i, [])
            sound_name = gap_matches[0].sound.name if gap_matches else "—"
            vol = volume_adjustments.get(i, 0.0)
            status = "[red]пропущен[/]" if i in skipped else "[green]активен[/]"

            table.add_row(
                str(i + 1),
                f"{format_time(gap.start_sec)} - {format_time(gap.end_sec)}",
                f"{gap.duration_sec:.2f}с",
                sound_name,
                f"{vol:+.1f} дБ",
                status,
            )

        console.print(table)
