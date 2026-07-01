from __future__ import annotations

import argparse
import json
import re

import torch
from PIL import Image, ImageOps
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, PretrainedConfig

from os import listdir
from os.path import isfile, join

DATA_DIR = "../../style_imgs/512"
OUT_PATH = "./florence_captions.jsonl"

MODEL_NAME = "microsoft/Florence-2-large"
TASK = "<CAPTION>"
STYLE_TOKEN = "<sks>"
NUM_BEAMS = 3  # use beamsearch



def clean_caption(text: str) -> str:
    text = text.lower()
    # get rid of starting descriptions that already incorporate the style as we're going to add it with the <sks> token later
    # e.g. "A cartoon of a girl" -> "a girl" 
    text = re.sub(
        r"^(an?|the)\s+(painting|drawing|illustration|cartoon|picture)\s+of\s+",
        "",
        text,
    )

    # get rid of adjectives that already contain style stuff
    text = re.sub(r"\b(painted|drawn|illustrated|cartoon|anime)\s+", "", text)
    
    # get rid of trailing dot, comma and whitespace
    text = text.strip(" ,.")

    return text 


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    image_paths = [DATA_DIR + "/" + f for f in listdir(DATA_DIR) if f[-4:] in [".jpg", ".png"]]
    if args.limit:
        image_paths = image_paths[: args.limit]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    # # Florence-2's bundled config predates forced_bos_token_id; add it so generate() works.
    # if not hasattr(PretrainedConfig, "forced_bos_token_id"):
    PretrainedConfig.forced_bos_token_id = None

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.additional_special_tokens = [] # expected by autoprocessor
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True, tokenizer=tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=dtype, trust_remote_code=True, attn_implementation="eager"
    )
    # Re-tie the shared embedding weights, which come apart when the model is loaded.
    shared_weight = model.language_model.model.shared.weight
    model.language_model.model.encoder.embed_tokens.weight = shared_weight
    model.language_model.model.decoder.embed_tokens.weight = shared_weight
    model.language_model.lm_head.weight = shared_weight
    model = model.to(device).eval()

    with open(OUT_PATH, "w", encoding="utf-8") as handle:
        for path in tqdm(image_paths):
            image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            inputs = processor(text=TASK, images=image, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            pixel_values = inputs["pixel_values"].to(device, dtype)

            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    max_new_tokens=40, # limit length
                    num_beams=3, # use beam search
                    do_sample=False,
                    use_cache=False,
                )

            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed = processor.post_process_generation(
                generated_text, task=TASK, image_size=(image.width, image.height)
            )
            raw_caption = parsed[TASK].strip()
            caption = clean_caption(raw_caption)
            row = {
                "image": path,
                "caption_raw": raw_caption,
                "caption": caption,
                "prompt": f"{caption}, in {STYLE_TOKEN} style",
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(image_paths)} captions to {OUT_PATH}")


if __name__ == "__main__":
    main()
