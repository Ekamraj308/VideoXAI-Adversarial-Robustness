"""
Dataset registry.

The paper evaluates three datasets independently (Med-M3, MedVideoCap-55K,
Kinetics-400) and compares medical-domain vs. general-domain robustness
(RQ2). This module makes that split explicit in code: each dataset gets
its own folder and its own "domain" label, and the pipeline processes and
reports results separately per dataset instead of pooling every video
into one flat folder.

Update DATASET_ROOTS to point at wherever each dataset's videos actually
live (e.g. subfolders under Google Drive). If a folder doesn't exist yet,
it's skipped with a warning rather than failing the whole run.
"""

from pathlib import Path

VIDEO_EXTS = [".mp4", ".webm", ".mkv", ".avi", ".mov"]

# name -> (folder path, domain label)
DATASET_ROOTS = {
    "Med-M3": {
        "path": str(Path(__file__).resolve().parent.parent / "data_local" / "med_m3"),
        "domain": "medical",
    },
    "MedVideoCap-55K": {
        "path": str(Path(__file__).resolve().parent.parent / "data_local" / "medvideocap_55k"),
        "domain": "medical",
    },
    "Kinetics-400": {
        "path": str(Path(__file__).resolve().parent.parent / "data_local" / "kinetics_400"),
        "domain": "general",
    },
}


def list_videos_for_dataset(dataset_name, limit=50):
    """Return sorted video file paths for one named dataset."""
    if dataset_name not in DATASET_ROOTS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    root = Path(DATASET_ROOTS[dataset_name]["path"])
    if not root.exists():
        print(f"Warning: dataset path does not exist, skipping: {root}")
        return []

    videos = [str(p) for p in root.glob("*") if p.suffix.lower() in VIDEO_EXTS]
    return sorted(videos)[:limit]


def all_datasets(limit=50):
    """
    Yield (dataset_name, domain, video_paths) for every configured dataset
    that actually has videos available.
    """
    for name, cfg in DATASET_ROOTS.items():
        videos = list_videos_for_dataset(name, limit=limit)
        if videos:
            yield name, cfg["domain"], videos
        else:
            print(f"Skipping {name}: no videos found at {cfg['path']}")
