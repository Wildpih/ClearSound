"""Configuration for ClearSound."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    """Application configuration."""

    # ── Paths ──────────────────────────────────────────────────────────
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    sounds_dir: Path = field(default=None)
    output_dir: Path = field(default=None)
    temp_dir: Path = field(default=None)

    # ── Audio Processing ───────────────────────────────────────────────
    sample_rate: int = 44100
    # demucs model: htdemucs (best quality), htdemucs_ft (fine-tuned), mdx_extra
    demucs_model: str = "htdemucs"
    # Number of parallel workers for demucs
    demucs_jobs: int = 2

    # ── Gap Analysis ───────────────────────────────────────────────────
    # Minimum gap duration (seconds) to consider for filling
    min_gap_duration: float = 0.3
    # Energy threshold (dB below max) to detect vocal presence
    vocal_energy_threshold_db: float = -30.0
    # Frame length for analysis (samples)
    analysis_frame_length: int = 2048
    # Hop length for analysis (samples)
    analysis_hop_length: int = 512

    # ── Sound Matching ─────────────────────────────────────────────────
    # Number of MFCC coefficients for spectral matching
    n_mfcc: int = 13
    # Number of top matches to consider
    top_matches: int = 5
    # Minimum similarity score (0.0 - 1.0) to accept a match
    min_similarity: float = 0.3

    # ── Sound Insertion ────────────────────────────────────────────────
    # Crossfade duration (seconds)
    crossfade_duration: float = 0.05
    # Maximum volume boost (dB) for inserted sounds
    max_boost_db: float = 6.0
    # Whether to apply spectral shaping to inserted sounds
    apply_spectral_shaping: bool = True

    # ── Interactive Preview ────────────────────────────────────────────
    # Preview context (seconds before/after each gap)
    preview_context: float = 2.0
    # Maximum iterations of user feedback
    max_feedback_iterations: int = 10

    # ── Freesound API (optional) ───────────────────────────────────────
    freesound_api_key: Optional[str] = None

    def __post_init__(self):
        import tempfile

        if self.sounds_dir is None:
            self.sounds_dir = self.project_root / "sounds"
        if self.output_dir is None:
            self.output_dir = self.project_root / "output"
        if self.temp_dir is None:
            local_temp = self.project_root / "temp"
            try:
                local_temp.mkdir(parents=True, exist_ok=True)
                # verify writable
                test_f = local_temp / ".write_test"
                test_f.touch()
                test_f.unlink()
                self.temp_dir = local_temp
            except Exception:
                # Fallback to user AppData/Temp directory if root directory is restricted
                fallback_temp = Path(tempfile.gettempdir()) / "ClearSound"
                fallback_temp.mkdir(parents=True, exist_ok=True)
                self.temp_dir = fallback_temp

        # Ensure directories exist
        try:
            self.sounds_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            self.sounds_dir = Path(tempfile.gettempdir()) / "ClearSound" / "sounds"
            self.sounds_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            self.output_dir = Path.home() / "ClearSound_Output"
            self.output_dir.mkdir(parents=True, exist_ok=True)

        if not self.temp_dir.exists():
            self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Read API key from environment
        if self.freesound_api_key is None:
            self.freesound_api_key = os.environ.get("FREESOUND_API_KEY")
