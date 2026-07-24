#!/usr/bin/env python

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


FILES = {
    "lora_out/pytorch_lora_weights.safetensors": "lora_out/pytorch_lora_weights.safetensors",
    "code/train_lora.py": "code/train_lora.py",
    "code/eval_lora.py": "code/eval_lora.py",
    "samples/adapter_00.png": "samples/adapter_00.png",
    "samples/adapter_01.png": "samples/adapter_01.png",
    "samples/adapter_02.png": "samples/adapter_02.png",
    "code/auto_captions/florence_captions.jsonl": "code/auto_captions/florence_captions.jsonl",
    "training_data/auxiliary.jsonl": "training_data/auxiliary.jsonl",
    "submission/requirements.txt": "requirements.txt",
    "submission/README.md": "README.md",
    "report.pdf": "report.pdf",
}


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
    for path in supplied_images + auxiliary_images:
        relative = path.relative_to(args.root).as_posix()
        files[relative] = relative

    missing = [source for source in files if not (args.root / source).is_file()]
    if missing:
        raise FileNotFoundError("Missing files: " + ", ".join(missing))

    with ZipFile(args.out, "w", compression=ZIP_DEFLATED) as archive:
        for source, destination in files.items():
            archive.write(args.root / source, destination)

    with ZipFile(args.out) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP verification failed")
    print(f"Created {args.out} with {len(files)} files")


if __name__ == "__main__":
    main()
