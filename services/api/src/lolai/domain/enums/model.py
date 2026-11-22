from enum import Enum

class Model(str, Enum):
    QWEN = "qwen3:8b"
    QWEN_EMBED = "qwen3-embedding:4b"
    QWEN_VL = "qwen2.5vl:7b"  # vision model, used to read augment cards from screenshots