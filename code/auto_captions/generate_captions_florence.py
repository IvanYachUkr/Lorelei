import argparse
import json
import os
import re
from pathlib import Path

import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoProcessor, Florence2ForConditionalGeneration


SCRIPT_DIR = Path(__file__).resolve().parent
ASSIGNMENT_DIR = SCRIPT_DIR.parents[1]
DATA_DIR = ASSIGNMENT_DIR / "style_imgs" / "512"
OUT_PATH = SCRIPT_DIR / "florence_captions.jsonl"
MODEL_NAME = "florence-community/Florence-2-base"


def clean_caption(text):
    text = text.lower()

    # get rid of starting descriptions that already incorporate the style because that's going to be added with the <sks> token later
    # e.g. "A cartoon of a girl" -> "a girl" 
    text = re.sub(
        r"^(an?|the)\s+(painting|drawing|illustration|cartoon|picture)\s+of\s+",
        "",
        text,
    )
    # get rid of adjectives that already contain style info
    text = re.sub(r"\b(painted|drawn|illustrated|cartoon|anime)\s+", "", text)
    
    # get rid of trailing dot, comma and whitespace
    text = text.strip(" ,.")

    # weird association the model has sometimes. It will just spit out "professor layton and the unwound future", which is a Nintendo DS-game with a similar-ish art style to ghibli
    # -> just label it "a scene" instead
    if "professor layton" in text:
        text = "a scene"

    return text 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_path", type=Path, default=OUT_PATH)
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    image_paths = sorted(
        path
        for path in DATA_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".png"}
    )
    if args.limit:
        image_paths = image_paths[: args.limit]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    model = Florence2ForConditionalGeneration.from_pretrained(
        args.model_name,
        dtype=dtype,
    ).to(device)
    processor = AutoProcessor.from_pretrained(args.model_name)

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