import json, random
import matplotlib.pyplot as plt
from PIL import Image

path = "auto_captions/florence_captions.jsonl"
rows = [json.loads(line) for line in open(path)]

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

random.seed(1233)
for ax, row in zip(axes.flat, random.sample(rows, 4)):
    ax.imshow(Image.open("auto_captions/" + row["image"]))
    ax.axis("off")
    ax.text(0.04, 0.04,
            row["caption"],
            transform=ax.transAxes,
            color="white",
            va="bottom",
            bbox=dict(facecolor="black", alpha=0.7),
            fontsize=12
            )

plt.tight_layout()
plt.savefig("caption_samples.png")
plt.show()