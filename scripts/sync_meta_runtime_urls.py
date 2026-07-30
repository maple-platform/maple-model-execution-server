#!/usr/bin/env python3
"""Synchronize AI_Models meta.json service URLs from models/*/config.yaml."""

import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_URLS = {
    "runtime-basic": "http://runtime-basic:8000",
    "runtime-medical": "http://runtime-medical:8000",
    "runtime-yolo": "http://runtime-yolo:8000",
    "runtime-nnunet": "http://runtime-nnunet:8000",
}
SERVICE_URL_PATTERN = re.compile(
    r'("service_url"\s*:\s*)"(?:[^"\\]|\\.)*"'
)


def main() -> None:
    meta_paths = {
        path.parent.name: path
        for path in ROOT.glob("AI_Models/*/*/meta.json")
    }
    changed = 0

    for config_path in sorted(ROOT.glob("models/*/config.yaml")):
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        model_name = config["model_name"]
        meta_path = meta_paths.get(model_name)
        if meta_path is None:
            # example_model intentionally has no registry metadata.
            if model_name != "example_model":
                raise RuntimeError(f"meta.json not found for {model_name}")
            continue

        expected_url = RUNTIME_URLS[config["runtime"]]
        text = meta_path.read_text(encoding="utf-8")
        data = json.loads(text)
        if "docker" not in data or "service_url" not in data["docker"]:
            raise RuntimeError(f"docker.service_url missing in {meta_path}")
        if data["docker"]["service_url"] == expected_url:
            continue

        updated, replacements = SERVICE_URL_PATTERN.subn(
            lambda match: f'{match.group(1)}"{expected_url}"',
            text,
            count=1,
        )
        if replacements != 1:
            raise RuntimeError(f"Could not update docker.service_url in {meta_path}")
        meta_path.write_text(updated, encoding="utf-8")
        changed += 1

    print(f"Updated {changed} meta.json files")


if __name__ == "__main__":
    main()
