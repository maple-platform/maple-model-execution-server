from __future__ import annotations

import base64
import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import SimpleITK as sitk


def predict(input_path: str, output_dir: str, config: dict) -> dict:
    name = str(config["model_name"])
    module_name = "_maple_msd_lung"
    if module_name in sys.modules:
        module = sys.modules[module_name]
    else:
        spec = importlib.util.spec_from_file_location(module_name, str(config["inference_path"]))
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    images, mask_image, measurements = module.main(input_path, str(config["model_path"]))
    slice_indices = measurements[0]["segmentation_slices"]
    if len(images) != len(slice_indices):
        raise RuntimeError("Image count does not match segmentation_slices")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = Path(input_path).name.replace(".nii.gz", "").replace(".nii", "")
    images_b64, image_metadata, output_files = [], [], []
    for image, slice_index in zip(images, slice_indices):
        buffer = io.BytesIO()
        Image.fromarray(image.astype(np.uint8), "RGB").save(buffer, format="PNG")
        raw = buffer.getvalue()
        image_path = output / f"{stem}_tumor_segmentation_slice_{slice_index:04d}.png"
        image_path.write_bytes(raw)
        images_b64.append(base64.b64encode(raw).decode("ascii"))
        image_metadata.append({"role": "segmentation_overlay_slice", "slice_index": slice_index})
        output_files.append(str(image_path))

    mask_path = output / f"{stem}_tumor_mask.nii.gz"
    sitk.WriteImage(mask_image, str(mask_path), True)
    output_files.append(str(mask_path))
    payload = {
        "model": name,
        "input": Path(input_path).name,
        "image_count": len(images),
        "labels": ["segmentation_overlay_slice"] * len(images),
        "image_metadata": image_metadata,
        "mask_file": mask_path.name,
        "mask_format": "nii.gz",
        "measurements": measurements,
    }
    result_path = output / f"{stem}_measurements.json"
    result_path.write_text(json.dumps(payload, indent=2) + "\n")
    output_files.append(str(result_path))
    return {**payload, "images_b64": images_b64, "output_files": output_files}
