import torch
from transformers import AutoTokenizer, Qwen3_5ForCausalLM, set_seed


def load_policy(config, adapter=None):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This experiment requires a CUDA GPU with BF16 support.")
    set_seed(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"], revision=config["model_revision"], padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model, info = Qwen3_5ForCausalLM.from_pretrained(
        config["model_id"], revision=config["model_revision"],
        dtype=torch.bfloat16, attn_implementation="sdpa", output_loading_info=True)
    if info.get("missing_keys") or info.get("mismatched_keys"):
        raise RuntimeError(f"Incomplete text model weights: {info}")
    model.to("cuda")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tokenizer
