#!/usr/bin/env python3
"""Build Mem-X's ETPNav runtime without editing the upstream checkout.

Only the Python standard library and git are needed to prepare the runtime.
Navigation still runs with the selected conda environment.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ADAPTER = Path(__file__).resolve().with_name("etpnav")
REPO = ADAPTER.parents[2]


def _normalized(path: Path) -> bytes:
    return (path.read_text().replace("\r\n", "\n").rstrip("\n") + "\n").encode()


def _link_tree(source: Path, target: Path) -> None:
    """Use real directories so replacing an adapted file cannot edit upstream."""
    target.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        if child.name in {".git", "__pycache__"}:
            continue
        dest = target / child.name
        if child.is_dir():
            _link_tree(child, dest)
        else:
            dest.symlink_to(child.resolve())


def prepare_runtime(source: Path, runtime_root: Path | None = None, *,
                    repo: Path = REPO, scenes_dir: Path | None = None,
                    scene_dataset_config: Path | None = None) -> Path:
    source = source.expanduser().resolve()
    repo = repo.expanduser().resolve()
    if (source / "memx_adapter.json").is_file():
        return source
    if not (source / "run.py").is_file():
        raise ValueError(f"ETPNav checkout is missing run.py: {source}")
    scenes = (scenes_dir or Path(os.environ.get("SCENES_DIR", str(repo / "scenes")))).expanduser().resolve()
    scene_config = (scene_dataset_config or Path(os.environ.get(
        "SCENE_DATASET_CONFIG", str(scenes / "hm3dsem/hm3d_annotated_basis.scene_dataset_config.json")
    ))).expanduser().resolve()
    manifest = json.loads((ADAPTER / "manifest.json").read_text())
    digest = hashlib.sha256()
    digest.update(json.dumps([str(source), str(repo), str(scenes), str(scene_config)]).encode())
    originals = {}
    for name, entry in manifest["files"].items():
        content = _normalized(source / name)
        sha = hashlib.sha256(content).hexdigest()
        if sha not in {entry["upstream_sha256"], entry["adapted_sha256"], entry.get("recovered_sha256")}:
            raise ValueError(
                f"Unsupported ETPNav source: {name}. Expected upstream commit "
                f"{manifest['upstream_commit']} or the recovered Mem-X adaptation; "
                "use a separate checkout at that commit. The source was not modified."
            )
        originals[name] = (content, sha)
        digest.update(content)
    for asset in sorted(ADAPTER.rglob("*")):
        if asset.is_file() and "__pycache__" not in asset.parts and asset.suffix in {".py", ".yaml", ".json", ".patch"}:
            digest.update(str(asset.relative_to(ADAPTER)).encode())
            digest.update(asset.read_bytes())
    digest.update(Path(__file__).read_bytes())
    root = (runtime_root or repo / "experiments_results/.runtime/etpnav").expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / digest.hexdigest()[:20]
    with (root / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (target / "memx_adapter.json").is_file():
            return target
        with tempfile.TemporaryDirectory(prefix=".prepare-", dir=root) as temporary:
            staging = Path(temporary) / "runtime"
            staging.mkdir()
            for child in source.iterdir():
                if child.name in {".git", "__pycache__", "data"}:
                    continue
                dest = staging / child.name
                if child.name in {"vlnce_baselines", "habitat_extensions", "run_r2r"}:
                    _link_tree(child, dest)
                else:
                    dest.symlink_to(child.resolve())
            for name, entry in manifest["files"].items():
                dest = staging / name
                dest.unlink()
                content, sha = originals[name]
                dest.write_bytes(content)
                patch_name = None
                if sha == entry["upstream_sha256"]:
                    patch_name = entry["patch"]
                elif sha == entry.get("recovered_sha256"):
                    patch_name = entry["upgrade_patch"]
                if patch_name:
                    result = subprocess.run(
                        ["git", "apply", "--no-index", str(ADAPTER / "patches" / patch_name)],
                        cwd=staging, text=True, capture_output=True,
                    )
                    if result.returncode:
                        raise RuntimeError(f"Could not apply {patch_name}: {result.stderr}")
                if hashlib.sha256(dest.read_bytes()).hexdigest() != entry["adapted_sha256"]:
                    raise RuntimeError(f"Adapted ETPNav file failed verification: {name}")
            protocol = ADAPTER / "protocol.py"
            if protocol.is_file():
                (staging / "memx_etpnav_protocol.py").write_bytes(protocol.read_bytes())
            # Model assets are shared; all generated logs belong to this runtime.
            data = staging / "data"
            data.mkdir()
            if (source / "data").is_dir():
                for child in (source / "data").iterdir():
                    if child.name != "logs":
                        (data / child.name).symlink_to(child.resolve())
            logs = data / "logs"
            logs.mkdir()
            checkpoints = source / "data/logs/checkpoints"
            (logs / "checkpoints").symlink_to(checkpoints)
            for name in ("running_log", "eval_results", "tensorboard_dirs", "video"):
                (logs / name).mkdir()
            for template in (ADAPTER / "configs").glob("*.yaml"):
                dest = staging / "run_r2r" / template.name
                if dest.is_symlink() or dest.exists():
                    dest.unlink()
                content = template.read_text()
                # JSON strings are also valid quoted YAML scalars.
                for token, value in {
                    "@SCENES_DIR@": str(scenes),
                    "@SCENE_DATASET_CONFIG@": str(scene_config),
                }.items():
                    content = content.replace(token, json.dumps(value))
                content = content.replace("@DATASET_DIR@", str(repo / "datasets"))
                dest.write_text(content)
            (staging / "memx_adapter.json").write_text(json.dumps({
                "source": str(source), "upstream_commit": manifest["upstream_commit"],
                "repo": str(repo), "scenes_dir": str(scenes),
            }, indent=2) + "\n")
            staging.rename(target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--scenes-dir", type=Path)
    parser.add_argument("--scene-dataset-config", type=Path)
    args = parser.parse_args()
    try:
        print(prepare_runtime(args.source, args.runtime_root, repo=args.repo,
                              scenes_dir=args.scenes_dir,
                              scene_dataset_config=args.scene_dataset_config))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"[etpnav-adapt] {exc}\n")


def adapt_arguments(arguments: list[str], level: int) -> list[str]:
    """Give direct baselines/run.py callers the same runtime as shell callers."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--etpnav-root", type=Path, default=Path(os.environ.get(
        "ETPNAV_ROOT", str(REPO / "baselines/models/ETPNav")
    )))
    parser.add_argument("--exp-config", type=Path)
    parser.add_argument("--task-config", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--scenes-dir", type=Path)
    args, _ = parser.parse_known_args(arguments)
    runtime = prepare_runtime(
        args.etpnav_root,
        args.output_root / ".runtime/etpnav" if args.output_root else None,
        scenes_dir=args.scenes_dir,
    )
    replacements = {"--etpnav-root": runtime}
    for option, value, default in (
        ("--exp-config", args.exp_config, f"run_r2r/iter_level{level}.yaml"),
        ("--task-config", args.task_config, f"run_r2r/hm3d_level{level}.yaml"),
    ):
        # Preserve custom config files, rebasing only the benchmark configs.
        if value is None or value.name in {Path(default).name}:
            replacements[option] = runtime / default
    result = []
    skip = False
    for item in arguments:
        if skip:
            skip = False
            continue
        key = item.split("=", 1)[0]
        if key in replacements:
            skip = "=" not in item
            continue
        result.append(item)
    for option, value in replacements.items():
        result.extend([option, str(value)])
    return result


if __name__ == "__main__":
    main()
