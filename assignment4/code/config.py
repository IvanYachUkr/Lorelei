from pathlib import Path


# ---------------------------------------------------------------------
# Model defaults
# ---------------------------------------------------------------------

DEFAULT_MODEL_NAME = "runwayml/stable-diffusion-v1-5"
DEFAULT_REVISION = None
DEFAULT_VARIANT = None


# ---------------------------------------------------------------------
# Custom token defaults
# ---------------------------------------------------------------------

DEFAULT_INSTANCE_TOKEN = "<sks>"
DEFAULT_TOKEN_INITIALIZER = "ghibli"


# ---------------------------------------------------------------------
# Prompt defaults
# ---------------------------------------------------------------------

DEFAULT_PROMPT_TEMPLATE = "an animated movie scene, in {instance_token} style"
DEFAULT_SAMPLE_PROMPT = f"a busy market, in {DEFAULT_INSTANCE_TOKEN} style"


# ---------------------------------------------------------------------
# Output / file format
# ---------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = Path("lora_out")
DEFAULT_SAMPLE_OUTDIR = Path("samples")

LORA_FILENAME = "pytorch_lora_weights.safetensors"
CUSTOM_TOKEN_EMBEDDING_KEY = "__custom_token_embedding__"


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


# ---------------------------------------------------------------------
# Training defaults
# ---------------------------------------------------------------------

DEFAULT_RANK = 8
DEFAULT_LORA_ALPHA = None

DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_ADAM_BETA1 = 0.9
DEFAULT_ADAM_BETA2 = 0.999
DEFAULT_ADAM_WEIGHT_DECAY = 1e-2
DEFAULT_ADAM_EPSILON = 1e-8
DEFAULT_MAX_GRAD_NORM = 1.0

DEFAULT_RESOLUTION = 512
DEFAULT_TRAIN_BATCH_SIZE = 1
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 4
DEFAULT_MAX_STEPS = 800
DEFAULT_LR_SCHEDULER = "constant"
DEFAULT_LR_WARMUP_STEPS = 0
DEFAULT_TRAIN_SEED = 42
DEFAULT_NUM_WORKERS = 0
DEFAULT_REPEATS = 1
DEFAULT_MIXED_PRECISION = "fp16"


# ---------------------------------------------------------------------
# Sampling / evaluation defaults
# ---------------------------------------------------------------------

DEFAULT_SAMPLE_NUM_IMAGES = 3
DEFAULT_SAMPLE_SEED = 1234
DEFAULT_NUM_INFERENCE_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_SAMPLE_HEIGHT = 512
DEFAULT_SAMPLE_WIDTH = 512
DEFAULT_SAMPLE_DTYPE = "auto"