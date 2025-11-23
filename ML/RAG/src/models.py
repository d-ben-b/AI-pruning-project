# src/models.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import requests
import json


# ========= HF LLM 後端 =========


def create_model(
    lm_model_name: str = "google/gemma-3-1b-it",
    device: str = None,
):
    """
    建立 HuggingFace chat LLM（預設用 gemma-3-1b-it）。
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(lm_model_name)

    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            lm_model_name,
            device_map="auto",
            dtype=torch.bfloat16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            lm_model_name,
            dtype=torch.float32,
        )
        model.to(device)

    model.eval()
    return model, tokenizer


def lm_template(
    text: str,
    system_prompt: str = "You are a helpful assistant.",
):
    """
    把問題包成 chat 格式，給 HF tokenizer.apply_chat_template 使用。
    """
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
):
    """
    HF 版本的文字產生。
    """
    inputs = tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    if hasattr(model, "device"):
        device = model.device
    else:
        device = next(model.parameters()).device

    for k, v in inputs.items():
        if v.dtype.is_floating_point:
            inputs[k] = v.to(device=device, dtype=model.dtype)
        else:
            inputs[k] = v.to(device)

    input_len = inputs["input_ids"].shape[-1]
    max_pos = getattr(model.config, "max_position_embeddings", None)
    if max_pos is not None and input_len > max_pos:
        raise ValueError(
            f"Input length {input_len} exceeds max_position_embeddings={max_pos}"
        )

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
    )[0]

    output_ids = output_ids[input_len:]
    text = tokenizer.decode(output_ids, skip_special_tokens=True)
    return text


def predict_no_rag(
    question: str,
    model,
    tokenizer,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
) -> str:
    """
    HF baseline：不使用 RAG。
    """
    prompt = lm_template(question)
    response = generate(
        prompt=prompt,
        tokenizer=tokenizer,
        model=model,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    return response.strip()


# ========= Ollama 後端 =========


def ollama_chat(
    messages,
    model_name: str = "llama3.2",
    base_url: str = "http://localhost:11434",
) -> str:
    """
    呼叫 Ollama 的 chat API：
    messages 為 list[{"role": ..., "content": ...}].
    """
    url = f"{base_url}/api/chat"
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": False,
    }
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    data = resp.json()
    # 回傳最後一則訊息內容
    return data["message"]["content"].strip()


def predict_no_rag_ollama(
    question: str,
    ollama_model: str = "llama3.2",
) -> str:
    """
    Ollama baseline：不使用 RAG。
    """
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": question,
        },
    ]
    return ollama_chat(messages, model_name=ollama_model)
