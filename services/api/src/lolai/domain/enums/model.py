from enum import Enum

class Model(str, Enum):
    QWEN = "qwen3:8b"
    QWEN_EMBED = "qwen3-embedding:4b"