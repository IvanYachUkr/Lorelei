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
    "submission/requirements.txt": "requirements.txt",
    "submission/README.md": "README.md",
    "report.pdf": "report.pdf",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("assignment4_submission.zip"))
    args = parser.parse_args()

    missing = [source for source in FILES if not (args.root / source).is_file()]
    if missing:
        raise FileNotFoundError("Missing files: " + ", ".join(missing))

    with ZipFile(args.out, "w", compression=ZIP_DEFLATED) as archive:
        for source, destination in FILES.items():
            archive.write(args.root / source, destination)

    with ZipFile(args.out) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP verification failed")
    print(f"Created {args.out} with {len(FILES)} files")


if __name__ == "__main__":
    main()
