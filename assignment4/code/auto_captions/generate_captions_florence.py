from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

# Florence's remote module lists flash_attn even when standard attention is used.
import transformers.dynamic_module_utils as _dyn
_orig_get_imports = _dyn.get_imports
_dyn.get_imports = lambda path: [m for m in _orig_get_imports(path) if m != "flash_attn"]


SCRIPT_DIR = Path(__file__).resolve().parent
ASSIGNMENT_DIR = SCRIPT_DIR.parents[1]
DATA_DIR = ASSIGNMENT_DIR / "style_imgs" / "512"
OUT_PATH = SCRIPT_DIR / "florence_captions.jsonl"
MODEL_NAME = "microsoft/Florence-2-base"


def clean_caption(text: str) -> str:
    text = text.lower()
    text = re.sub(
        r"^(an?|the)\s+(painting|drawing|illustration|cartoon|picture)\s+of\s+",
        "",
        text,
    )
    text = re.sub(r"\b(painted|drawn|illustrated|cartoon|anime)\s+", "", text)
    text = text.strip(" ,.")
    if "professor layton" in text:
        text = "a scene"
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out_path", type=Path, default=OUT_PATH)
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    image_paths = sorted(
        path
        for path in args.data_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".png"}
    )
    if args.limit:
        image_paths = image_paths[: args.limit]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)

    task = "<CAPTION>"

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    with args.out_path.open("w", encoding="utf-8") as handle:
        for path in tqdm(image_paths):
            image = Image.open(path)
            inputs = processor(text=task, images=image, return_tensors="pt").to(device, dtype)
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                do_sample=False,
                num_beams=3,
            )

            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = processor.post_process_generation(
                generated_text,
                task=task,
                image_size=(image.width, image.height),
            )

            output_txt = parsed_answer[task]
            caption = clean_caption(output_txt)

            row = {
                "image": os.path.relpath(path, args.out_path.parent).replace("\\", "/"),
                "caption_raw": parsed_answer,
                "caption": caption,
                "prompt": f"{caption}, in <sks> style",
            }
            handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
