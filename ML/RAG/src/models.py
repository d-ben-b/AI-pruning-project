# src/models.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any


def create_model(
    lm_model_name: str = "google/gemma-3-1b-it",
):
    tokenizer = AutoTokenizer.from_pretrained(lm_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        lm_model_name,
        device_map="auto",
    ).eval()
    return model, tokenizer


def lm_template(
    text: str,
    system_prompt: str = (
        "You are a concise machine learning tutor. "
        "Answer in 1–3 sentences and focus on giving a clear definition."
    ),
) -> List[Dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": text}],
        },
    ]


@torch.inference_mode()
def generate(
    prompt,
    tokenizer,
    model,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
) -> str:
    inputs = tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {
        k: (
            v.to(model.device, dtype=model.dtype)
            if v.dtype.is_floating_point
            else v.to(model.device)
        )
        for k, v in inputs.items()
    }

    input_len = inputs["input_ids"].shape[-1]
    max_len = int(getattr(model.config, "max_position_embeddings", 8192))
    if input_len > max_len:
        raise ValueError(
            f"Input length {input_len} exceeds maximum allowed length of {max_len} tokens."
        )

    generation = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
    )
    generation = generation[0][input_len:]
    return tokenizer.decode(generation, skip_special_tokens=True).strip()


def predict_no_rag(
    question: str,
    model,
    tokenizer,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
) -> str:
    prompt = lm_template(question)
    return generate(
        prompt=prompt,
        tokenizer=tokenizer,
        model=model,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
