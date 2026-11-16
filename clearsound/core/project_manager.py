"""Multi-Project Manager and Hub for ClearSound.

Provides persistence, serialization, switching, and management for
multiple video/audio restoration projects.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from clearsound.config import Config
from clearsound.core.audio_utils import format_time, get_waveform_peaks, load_audio
from clearsound.core.gap_analyzer import GapInfo
from clearsound.core.sound_matcher import MatchResult
from clearsound.library.sound_library import SoundEntry


@dataclass
class Project:
    """A ClearSound project representing an audio/video file and its restoration state."""

    id: str
    name: str
    source_path: Path
    is_video: bool = False
    status: str = "new"  # "new", "processed", "processing", "error"
    duration_sec: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M"))
    updated_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M"))

    # Audio paths
    no_vocals_wav: Optional[Path] = None
    enhanced_wav: Optional[Path] = None
    preview_video_path: Optional[Path] = None
    preview_novocal_video_path: Optional[Path] = None

    # Restoration state
    gaps: List[GapInfo] = field(default_factory=list)
    matches: Dict[int, List[MatchResult]] = field(default_factory=dict)
    volume_adjustments: Dict[int, float] = field(default_factory=dict)
    sound_overrides: Dict[int, Path] = field(default_factory=dict)
    enabled_map: Dict[int, bool] = field(default_factory=dict)

    # Cached visual waveform
    waveform_peaks: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def formatted_duration(self) -> str:
        return format_time(self.duration_sec) if self.duration_sec > 0 else "00:00"

    @property
    def gaps_count(self) -> int:
        return len(self.gaps)

    def mark_updated(self) -> None:
        self.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M")


class ProjectManager:
    """Manages project persistence and workspace directory."""

    def __init__(self, config: Config):
        self.config = config
        self.projects_dir = config.projects_dir
        self.projects_dir.mkdir(parents=True, exist_ok=True)

    def _sanitize_id(self, name: str) -> str:
        clean = re.sub(r"[^a-zA-Z0-9_\-]", "_", name).lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"proj_{timestamp}_{clean[:20]}"

    def create_project(self, source_path: Path | str, name: Optional[str] = None) -> Project:
        """Create a new project from a media file and save its initial state."""
        source_path = Path(source_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Файл не найден: {source_path}")

        project_name = name or source_path.name
        project_id = self._sanitize_id(source_path.stem)
        video_exts = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}
        is_video = source_path.suffix.lower() in video_exts

        proj_folder = self.projects_dir / project_id
        proj_folder.mkdir(parents=True, exist_ok=True)

        project = Project(
            id=project_id,
            name=project_name,
            source_path=source_path,
            is_video=is_video,
            status="new",
        )

        # Quick probe of duration and waveform
        try:
            audio, sr = load_audio(source_path, sr=self.config.sample_rate)
            samples = audio.shape[1] if audio.ndim > 1 else len(audio)
            project.duration_sec = samples / sr
            project.waveform_peaks = get_waveform_peaks(audio, num_points=1200)
        except Exception:
            pass

        self.save_project(project)
        return project

    def save_project(self, project: Project) -> None:
        """Save project metadata and features to disk."""
        proj_folder = self.projects_dir / project.id
        proj_folder.mkdir(parents=True, exist_ok=True)

        project.mark_updated()

        # Serialize gaps
        serialized_gaps = []
        mfcc_list = []
        spec_env_list = []

        for gap in project.gaps:
            gap_dict = {
                "start_sample": gap.start_sample,
                "end_sample": gap.end_sample,
                "sample_rate": gap.sample_rate,
                "avg_rms": gap.avg_rms,
                "spectral_centroid": gap.spectral_centroid,
                "category_hint": gap.category_hint,
                "energy_loss_ratio": gap.energy_loss_ratio,
            }
            serialized_gaps.append(gap_dict)
            mfcc_list.append(gap.mfcc_profile if gap.mfcc_profile is not None else np.array([]))
            spec_env_list.append(gap.spectral_envelope if gap.spectral_envelope is not None else np.array([]))

        # Serialize matches
        serialized_matches = {}
        for gap_idx, match_list in project.matches.items():
            serialized_matches[str(gap_idx)] = [
                {
                    "sound_path": str(m.sound.path),
                    "sound_name": m.sound.name,
                    "category": m.sound.category,
                    "similarity_score": m.similarity_score,
                    "category_match": m.category_match,
                    "spectral_distance": m.spectral_distance,
                    "recommended_volume_db": m.recommended_volume_db,
                }
                for m in match_list
            ]

        # Serialized overrides and volume adjustments
        overrides_dict = {str(k): str(v) for k, v in project.sound_overrides.items()}
        vol_dict = {str(k): v for k, v in project.volume_adjustments.items()}
        enabled_dict = {str(k): v for k, v in project.enabled_map.items()}

        meta = {
            "id": project.id,
            "name": project.name,
            "source_path": str(project.source_path),
            "is_video": project.is_video,
            "status": project.status,
            "duration_sec": project.duration_sec,
            "created_at": project.created_at,
            "updated_at": project.updated_at,
            "no_vocals_wav": str(project.no_vocals_wav) if project.no_vocals_wav else None,
            "enhanced_wav": str(project.enhanced_wav) if project.enhanced_wav else None,
            "preview_video_path": str(project.preview_video_path) if project.preview_video_path else None,
            "preview_novocal_video_path": str(project.preview_novocal_video_path) if project.preview_novocal_video_path else None,
            "gaps": serialized_gaps,
            "matches": serialized_matches,
            "sound_overrides": overrides_dict,
            "volume_adjustments": vol_dict,
            "enabled_map": enabled_dict,
        }

        # Write project.json
        meta_file = proj_folder / "project.json"
        meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

        # Save binary arrays (waveform peaks)
        if project.waveform_peaks is not None:
            np.save(str(proj_folder / "waveform.npy"), project.waveform_peaks)

    def load_project(self, project_id: str) -> Optional[Project]:
        """Load a project by ID from disk."""
        proj_folder = self.projects_dir / project_id
        meta_file = proj_folder / "project.json"
        if not meta_file.exists():
            return None

        try:
            data = json.loads(meta_file.read_text(encoding="utf-8"))

            project = Project(
                id=data["id"],
                name=data["name"],
                source_path=Path(data["source_path"]),
                is_video=data.get("is_video", False),
                status=data.get("status", "new"),
                duration_sec=data.get("duration_sec", 0.0),
                created_at=data.get("created_at", ""),
                updated_at=data.get("updated_at", ""),
                no_vocals_wav=Path(data["no_vocals_wav"]) if data.get("no_vocals_wav") else None,
                enhanced_wav=Path(data["enhanced_wav"]) if data.get("enhanced_wav") else None,
                preview_video_path=Path(data["preview_video_path"]) if data.get("preview_video_path") else None,
                preview_novocal_video_path=Path(data["preview_novocal_video_path"]) if data.get("preview_novocal_video_path") else None,
            )

            # Reconstruct gaps
            gaps = []
            for g in data.get("gaps", []):
                gap = GapInfo(
                    start_sample=g["start_sample"],
                    end_sample=g["end_sample"],
                    sample_rate=g["sample_rate"],
                    avg_rms=g.get("avg_rms", 0.0),
                    spectral_centroid=g.get("spectral_centroid", 0.0),
                    category_hint=g.get("category_hint", "ambient"),
                    energy_loss_ratio=g.get("energy_loss_ratio", 0.0),
                )
                gaps.append(gap)
            project.gaps = gaps

            # Reconstruct matches
            matches = {}
            for k, match_list in data.get("matches", {}).items():
                m_list = []
                for m in match_list:
                    sound = SoundEntry(
                        path=Path(m["sound_path"]),
                        name=m["sound_name"],
                        category=m["category"],
                    )
                    m_res = MatchResult(
                        sound=sound,
                        similarity_score=m.get("similarity_score", 0.8),
                        category_match=m.get("category_match", True),
                        spectral_distance=m.get("spectral_distance", 0.5),
                        recommended_volume_db=m.get("recommended_volume_db", 0.0),
                    )
                    m_list.append(m_res)
                matches[int(k)] = m_list
            project.matches = matches

            # Reconstruct overrides
            project.volume_adjustments = {int(k): float(v) for k, v in data.get("volume_adjustments", {}).items()}
            project.sound_overrides = {int(k): Path(v) for k, v in data.get("sound_overrides", {}).items()}
            project.enabled_map = {int(k): bool(v) for k, v in data.get("enabled_map", {}).items()}

            # Load waveform if present
            wave_path = proj_folder / "waveform.npy"
            if wave_path.exists():
                project.waveform_peaks = np.load(str(wave_path))

            return project
        except Exception as e:
            print(f"Error loading project {project_id}: {e}")
            return None

    def list_projects(self) -> List[Project]:
        """List all available projects sorted by latest update."""
        projects = []
        if not self.projects_dir.exists():
            return []

        for p_dir in self.projects_dir.iterdir():
            if p_dir.is_dir() and (p_dir / "project.json").exists():
                proj = self.load_project(p_dir.name)
                if proj:
                    projects.append(proj)

        # Sort updated_at descending
        projects.sort(key=lambda p: p.updated_at, reverse=True)
        return projects

    def delete_project(self, project_id: str) -> bool:
        """Delete a project and all its cached files."""
        proj_folder = self.projects_dir / project_id
        if proj_folder.exists():
            shutil.rmtree(proj_folder, ignore_errors=True)
            return True
        return False

    def rename_project(self, project_id: str, new_name: str) -> bool:
        """Rename an existing project."""
        proj = self.load_project(project_id)
        if proj:
            proj.name = new_name.strip()
            self.save_project(proj)
            return True
        return False
