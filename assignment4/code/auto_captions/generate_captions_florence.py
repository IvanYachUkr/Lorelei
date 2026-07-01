from __future__ import annotations

import argparse
import json
import re

import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

from os import listdir

# dumb dependency issue with flash_attn module -> make it not use it
import transformers.dynamic_module_utils as _dyn
_orig_get_imports = _dyn.get_imports
_dyn.get_imports = lambda path: [m for m in _orig_get_imports(path) if m != "flash_attn"]


DATA_DIR = "../../style_imgs/512"
OUT_PATH = "./florence_captions.jsonl"
MODEL_NAME = "microsoft/Florence-2-large"

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
    return text 

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    image_paths = [DATA_DIR + "/" + f for f in listdir(DATA_DIR) if f[-4:] in [".jpg", ".png"]]
    if args.limit:
        image_paths = image_paths[: args.limit]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", torch_dtype=dtype, trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)


    TASK="<CAPTION>"

    with open(OUT_PATH, "w", encoding="utf-8") as handle:
        for path in tqdm(image_paths):

            image = Image.open(path)
            inputs = processor(text=TASK, images=image, return_tensors="pt").to(device, dtype)
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                do_sample=False,
                num_beams=3,
            )

            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = processor.post_process_generation(generated_text, task=TASK, image_size=(image.width, image.height))

            output_txt = parsed_answer[TASK]
            caption = clean_caption(output_txt)

            row = {
                "image": path,
                "caption_raw": parsed_answer,
                "caption": caption,
                "prompt": f"{caption}, in <sks> style",
            }
            handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
