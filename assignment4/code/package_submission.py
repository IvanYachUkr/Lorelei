#!/usr/bin/env python

import argparse
import json
import unicodedata
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile


FILES = {
    "lora_out/pytorch_lora_weights.safetensors": "lora_out/pytorch_lora_weights.safetensors",
    "code/train_lora.py": "code/train_lora.py",
    "code/eval_lora.py": "code/eval_lora.py",
    "samples/adapter_00.png": "samples/adapter_00.png",
    "samples/adapter_01.png": "samples/adapter_01.png",
    "samples/adapter_02.png": "samples/adapter_02.png",
    "training_data/auxiliary.jsonl": "training_data/auxiliary.jsonl",
    "submission/requirements.txt": "requirements.txt",
    "submission/README.md": "README.md",
    "report.pdf": "report.pdf",
}

CAPTIONS = "code/auto_captions/florence_captions.jsonl"


def portable_filename(name):
    normalized = unicodedata.normalize("NFKD", name)
    return "".join(char for char in normalized if ord(char) < 128 and not unicodedata.combining(char))


def portable_captions(path, supplied_names):
    rows = []
    caption_names = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            name = portable_filename(PurePosixPath(row["image"]).name)
            row["image"] = f"../../style_imgs/512/{name}"
            rows.append(row)
            caption_names.append(name)
    if len(rows) != 843 or set(caption_names) != supplied_names:
        raise ValueError("Caption paths do not match the supplied images")
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("assignment4_submission.zip"))
    args = parser.parse_args()

    files = dict(FILES)
    supplied_images = sorted((args.root / "style_imgs/512").glob("*"))
    auxiliary_images = sorted((args.root / "training_data/images").glob("*.png"))
    if len(supplied_images) != 843 or len(auxiliary_images) != 60:
        raise ValueError("Expected 843 supplied images and 60 auxiliary images")
    supplied_names = {portable_filename(path.name) for path in supplied_images}
    if len(supplied_names) != len(supplied_images):
        raise ValueError("Portable supplied-image names are not unique")
    for path in supplied_images:
        relative = path.relative_to(args.root).as_posix()
        files[relative] = f"style_imgs/512/{portable_filename(path.name)}"
    for path in auxiliary_images:
        relative = path.relative_to(args.root).as_posix()
        files[relative] = relative

    missing = [source for source in [*files, CAPTIONS] if not (args.root / source).is_file()]
    if missing:
        raise FileNotFoundError("Missing files: " + ", ".join(missing))

    with ZipFile(args.out, "w", compression=ZIP_DEFLATED) as archive:
        for source, destination in files.items():
            archive.write(args.root / source, destination)
        archive.writestr(CAPTIONS, portable_captions(args.root / CAPTIONS, supplied_names))

    with ZipFile(args.out) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP verification failed")
    print(f"Created {args.out} with {len(files) + 1} files")


if __name__ == "__main__":
    main()
