"""Voice removal module using Demucs.

Separates audio into stems (vocals, drums, bass, other) and returns
the combined non-vocal stems as the cleaned track.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from clearsound.config import Config
from clearsound.core.audio_utils import load_audio, save_audio

console = Console()


class VoiceRemover:
    """Remove vocals from audio using Demucs source separation."""

    def __init__(self, config: Config):
        self.config = config
        self._separated_dir = config.temp_dir / "separated"
        self._separated_dir.mkdir(parents=True, exist_ok=True)

    def separate(self, input_path: str | Path
                 ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Run full vocal separation.

        Args:
            input_path: Path to the audio file (WAV, MP3, FLAC, etc.).

        Returns:
            (no_vocals, vocals, sample_rate)
            - no_vocals: combined non-vocal stems, shape (channels, samples).
            - vocals: the vocal stem, shape (channels, samples).
            - sample_rate: sample rate of the output.
        """
        input_path = Path(input_path)
        console.print(f"\n[bold cyan]🎵 Разделяем аудио на стемы "
                      f"(модель: {self.config.demucs_model})...[/]")

        # ── Prepare a WAV for demucs if the input is a video ──────────
        from clearsound.core.audio_utils import extract_audio_from_video
        video_exts = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}
        if input_path.suffix.lower() in video_exts:
            wav_path = self.config.temp_dir / (input_path.stem + "_extracted.wav")
            if not wav_path.exists():
                console.print("[dim]  Извлекаем аудио из видео...[/]")
                extract_audio_from_video(input_path, wav_path, self.config.sample_rate)
            process_path = wav_path
        else:
            process_path = input_path

        # ── Run Demucs ────────────────────────────────────────────────
        output_dir = self._separated_dir / "demucs_out"
        if output_dir.exists():
            shutil.rmtree(output_dir)

        cmd = [
            "python", "-m", "demucs",
            "--two-stems", "vocals",     # only separate vocals vs rest
            "-n", self.config.demucs_model,
            "-o", str(output_dir),
            "--jobs", str(self.config.demucs_jobs),
            str(process_path),
        ]

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("  Demucs обрабатывает...", total=None)
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )
            progress.update(task, completed=True)

        if result.returncode != 0:
            console.print(f"[bold red]Ошибка Demucs:[/]\n{result.stderr}")
            raise RuntimeError(f"Demucs failed: {result.stderr}")

        # ── Load separated stems ──────────────────────────────────────
        model_dir = output_dir / self.config.demucs_model / process_path.stem

        no_vocals_path = model_dir / "no_vocals.wav"
        vocals_path = model_dir / "vocals.wav"

        if not no_vocals_path.exists() or not vocals_path.exists():
            # Demucs may name them differently
            available = list(model_dir.glob("*.wav"))
            console.print(f"[dim]  Доступные стемы: {[f.name for f in available]}[/]")
            # Try alternative names
            for f in available:
                if "vocal" in f.name.lower() and "no" not in f.name.lower():
                    vocals_path = f
                elif "no_vocal" in f.name.lower() or "other" in f.name.lower():
                    no_vocals_path = f

        no_vocals, sr = load_audio(no_vocals_path, sr=self.config.sample_rate)
        vocals, _ = load_audio(vocals_path, sr=self.config.sample_rate)

        console.print("[bold green]✓ Голос успешно отделён![/]")
        return no_vocals, vocals, sr

    def remove_voice(self, input_path: str | Path,
                     output_path: Optional[str | Path] = None
                     ) -> Tuple[Path, np.ndarray, np.ndarray, int]:
        """Convenience method: separate and save the no-vocals track.

        Returns:
            (saved_path, no_vocals, vocals, sample_rate)
        """
        no_vocals, vocals, sr = self.separate(input_path)

        if output_path is None:
            input_path = Path(input_path)
            output_path = self.config.output_dir / f"{input_path.stem}_no_vocals.wav"

        saved = save_audio(no_vocals, output_path, sr)
        console.print(f"[dim]  Сохранено: {saved}[/]")
        return saved, no_vocals, vocals, sr
