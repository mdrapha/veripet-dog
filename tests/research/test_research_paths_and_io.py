import csv
import json
from pathlib import Path

from veripet.research.config import PathsConfig
from veripet.research.io import save_checkpoint_bytes, save_csv, save_json


def test_paths_and_artifacts_follow_species_task_tree(tmp_path: Path):
    paths = PathsConfig(
        project_root=tmp_path,
        drive_root=Path("drive/MyDrive/veripet"),
        species="dog",
        task="verification",
        experiment_name="arcface_baseline",
    )
    paths.ensure_tree()

    save_json({"seed": 42}, paths.results_dir() / "config.json")
    save_csv([{"metric": "auc", "value": 0.9}], paths.results_dir() / "metrics.csv")
    save_checkpoint_bytes(b"abc", paths.artifacts_dir() / "checkpoint.pth")

    assert "dogs/verification" in str(paths.research_root())
    assert (paths.results_dir() / "config.json").exists()
    assert (paths.results_dir() / "metrics.csv").exists()
    assert (paths.artifacts_dir() / "checkpoint.pth").exists()
