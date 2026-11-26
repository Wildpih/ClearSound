"""ClearSound — main entry point and CLI.

Orchestrates the full pipeline:
1. Extract audio / load file
2. Remove vocals (Demucs)
3. Analyze gaps where voice was
4. Search for matching sounds
5. Insert sounds and enhance
6. Interactive preview & correction
7. Save final result
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click
import numpy as np
from rich.console import Console
from rich.panel import Panel

from clearsound import __version__
from clearsound.config import Config
from clearsound.core.audio_utils import load_audio, save_audio, format_time, samples_to_seconds
from clearsound.core.gap_analyzer import GapAnalyzer
from clearsound.core.sound_inserter import SoundInserter
from clearsound.core.sound_matcher import SoundMatcher
from clearsound.core.voice_remover import VoiceRemover
from clearsound.library.sound_library import SoundLibrary
from clearsound.ui.interactive import InteractivePreview

console = Console()


def print_banner() -> None:
    """Print the application banner."""
    banner = f"""
[bold cyan]
   ██████╗██╗     ███████╗ █████╗ ██████╗ 
  ██╔════╝██║     ██╔════╝██╔══██╗██╔══██╗
  ██║     ██║     █████╗  ███████║██████╔╝
  ██║     ██║     ██╔══╝  ██╔══██║██╔══██╗
  ╚██████╗███████╗███████╗██║  ██║██║  ██║
   ╚═════╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝
  ███████╗ ██████╗ ██╗   ██╗███╗   ██╗██████╗ 
  ██╔════╝██╔═══██╗██║   ██║████╗  ██║██╔══██╗
  ███████╗██║   ██║██║   ██║██╔██╗ ██║██║  ██║
  ╚════██║██║   ██║██║   ██║██║╚██╗██║██║  ██║
  ███████║╚██████╔╝╚██████╔╝██║ ╚████║██████╔╝
  ╚══════╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═══╝╚═════╝ 
[/]
  [dim]v{__version__} — Удаление голоса и восстановление звуковой дорожки[/]
"""
    console.print(banner)


def run_pipeline(
    input_path: str,
    output_path: Optional[str] = None,
    config: Optional[Config] = None,
    interactive: bool = True,
    generate_sounds: bool = True,
) -> Path:
    """Run the full ClearSound pipeline.

    Args:
        input_path: Path to the input audio or video file.
        output_path: Where to save the result. Auto-generated if None.
        config: Configuration object. Uses defaults if None.
        interactive: Whether to run the interactive preview.
        generate_sounds: Whether to generate basic sounds if library is empty.

    Returns:
        Path to the saved output file.
    """
    if config is None:
        config = Config()

    input_file = Path(input_path)
    if not input_file.exists():
        console.print(f"[bold red]❌ Файл не найден: {input_file}[/]")
        sys.exit(1)

    if output_path is None:
        output_path = str(
            config.output_dir / f"{input_file.stem}_clearsound.wav"
        )

    print_banner()
    console.print(Panel(
        f"[bold]Входной файл:[/] {input_file}\n"
        f"[bold]Выходной файл:[/] {output_path}\n"
        f"[bold]Модель Demucs:[/] {config.demucs_model}",
        title="📋 Параметры обработки",
    ))

    # ── Step 0: Load original audio ───────────────────────────────────
    console.print("\n[bold]═══ Шаг 0: Загрузка аудио ═══[/]")
    original, sr = load_audio(input_file, sr=config.sample_rate)
    duration = samples_to_seconds(
        original.shape[1] if original.ndim > 1 else len(original), sr
    )
    console.print(
        f"  Длительность: {format_time(duration)}, "
        f"Каналов: {original.shape[0] if original.ndim > 1 else 1}, "
        f"Частота: {sr} Hz"
    )

    # ── Step 1: Remove vocals ─────────────────────────────────────────
    console.print("\n[bold]═══ Шаг 1: Удаление голоса ═══[/]")
    remover = VoiceRemover(config)
    no_vocals, vocals, sr = remover.separate(input_file)

    # ── Step 2: Analyze gaps ──────────────────────────────────────────
    console.print("\n[bold]═══ Шаг 2: Анализ звуковых пробелов ═══[/]")
    analyzer = GapAnalyzer(config)
    gaps, sound_events, speech_regions = analyzer.analyze(original, no_vocals, vocals, sr)

    if not gaps:
        console.print("[yellow]Голосовых сегментов не найдено. "
                      "Сохраняем аудио без голоса как есть.[/]")
        saved = save_audio(no_vocals, output_path, sr)
        console.print(f"[bold green]💾 Сохранено: {saved}[/]")
        return saved

    # ── Step 3: Search for matching sounds ────────────────────────────
    console.print("\n[bold]═══ Шаг 3: Поиск подходящих звуков ═══[/]")
    library = SoundLibrary(config)

    # Generate basic sounds if library is empty
    if library.count() == 0 and generate_sounds:
        console.print("[yellow]Библиотека пуста. Генерируем базовые звуки...[/]")
        n = library.generate_basic_sounds()
        console.print(f"[green]  Сгенерировано {n} звуков[/]")

    library.print_summary()

    matcher = SoundMatcher(config, library)
    matches = matcher.find_matches(gaps)

    # ── Step 4: Insert sounds ─────────────────────────────────────────
    console.print("\n[bold]═══ Шаг 4: Вставка звуков ═══[/]")
    inserter = SoundInserter(config)
    enhanced = inserter.insert(no_vocals, gaps, matches, sr, vocals=vocals, original=original)

    # ── Step 5: Interactive preview ───────────────────────────────────
    if interactive:
        console.print("\n[bold]═══ Шаг 5: Предпрослушивание ═══[/]")
        preview = InteractivePreview(config)
        final_audio, vol_adj = preview.run(
            original, no_vocals, enhanced, gaps, matches, sr
        )

        # If the user made adjustments, re-run insertion
        if vol_adj:
            console.print("\n[dim]Применяем коррективы...[/]")
            final_audio = inserter.insert(
                no_vocals, gaps, matches, sr, volume_adjustments=vol_adj, vocals=vocals, original=original
            )
    else:
        final_audio = enhanced

    # ── Save result ───────────────────────────────────────────────────
    saved = save_audio(final_audio, output_path, sr)
    console.print(Panel(
        f"[bold green]✅ Готово![/]\n\n"
        f"Файл сохранён: [bold]{saved}[/]\n"
        f"Обработано сегментов: {len(gaps)}\n"
        f"Длительность: {format_time(duration)}",
        title="🎉 Результат",
    ))

    return saved


# ── CLI Interface ─────────────────────────────────────────────────────

@click.group(invoke_without_command=True)
@click.pass_context
@click.version_option(__version__)
def cli(ctx):
    """ClearSound — удаление голоса и восстановление звуковой дорожки."""
    if ctx.invoked_subcommand is None:
        from clearsound.ui.app import run_app
        run_app()


@cli.command()
@click.argument("input_file", type=click.Path(exists=True), required=False)
def gui(input_file):
    """Запустить графический интерфейс (UI)."""
    from clearsound.ui.app import run_app
    run_app(input_file)


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), default=None,
              help="Путь к выходному файлу.")
@click.option("--model", default="htdemucs",
              help="Модель Demucs (htdemucs, htdemucs_ft, mdx_extra).")
@click.option("--no-interactive", is_flag=True,
              help="Пропустить интерактивный просмотр.")
@click.option("--no-generate", is_flag=True,
              help="Не генерировать базовые звуки если библиотека пуста.")
@click.option("--min-gap", type=float, default=0.3,
              help="Минимальная длительность пробела (секунды).")
@click.option("--crossfade", type=float, default=0.05,
              help="Длительность кроссфейда (секунды).")
def process(input_file, output, model, no_interactive, no_generate,
            min_gap, crossfade):
    """Обработать аудио/видео файл: удалить голос и восстановить звук."""
    config = Config()
    config.demucs_model = model
    config.min_gap_duration = min_gap
    config.crossfade_duration = crossfade

    run_pipeline(
        input_path=input_file,
        output_path=output,
        config=config,
        interactive=not no_interactive,
        generate_sounds=not no_generate,
    )


@cli.command()
@click.argument("sound_file", type=click.Path(exists=True))
@click.option("-c", "--category", default="foley",
              type=click.Choice(["ambient", "music", "nature", "urban", "foley"]),
              help="Категория звука.")
def add_sound(sound_file, category):
    """Добавить звук в библиотеку."""
    config = Config()
    library = SoundLibrary(config)
    entry = library.add_sound(sound_file, category)
    console.print(f"[green]✓ Добавлен: {entry.name} (категория: {entry.category})[/]")


@cli.command()
def library():
    """Показать содержимое библиотеки звуков."""
    config = Config()
    lib = SoundLibrary(config)
    lib.print_summary()


@cli.command()
def generate():
    """Сгенерировать базовые синтетические звуки."""
    config = Config()
    lib = SoundLibrary(config)
    n = lib.generate_basic_sounds()
    console.print(f"[green]✓ Сгенерировано {n} новых звуков[/]")
    lib.print_summary()


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), default=None,
              help="Путь к выходному файлу.")
@click.option("--model", default="htdemucs",
              help="Модель Demucs.")
def remove_vocals(input_file, output, model):
    """Только удалить голос (без восстановления)."""
    config = Config()
    config.demucs_model = model
    remover = VoiceRemover(config)
    remover.remove_voice(input_file, output)


if __name__ == "__main__":
    cli()
