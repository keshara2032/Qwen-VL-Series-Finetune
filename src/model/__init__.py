from .load_model import (
    apply_qwen_vl_monkey_patches,
    load_qwen_vl_generation_model,
    load_qwen_vl_sequence_classification_model,
)

__all__ = [
    "apply_qwen_vl_monkey_patches",
    "load_qwen_vl_generation_model",
    "load_qwen_vl_sequence_classification_model",
    "Qwen2VLForSequenceClassification",
    "Qwen2_5_VLForSequenceClassification",
    "Qwen3VLForSequenceClassification",
    "Qwen3_5ForSequenceClassification",
    "Qwen3_5MoeForSequenceClassification",
]


def __getattr__(name):
    if name in {
        "Qwen2VLForSequenceClassification",
        "Qwen2_5_VLForSequenceClassification",
        "Qwen3VLForSequenceClassification",
        "Qwen3_5ForSequenceClassification",
        "Qwen3_5MoeForSequenceClassification",
    }:
        from . import modeling_cls

        return getattr(modeling_cls, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
