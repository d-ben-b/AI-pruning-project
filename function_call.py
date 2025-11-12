# 安裝必要套件
# pip install -U qwen-agent

import json
from qwen_agent.llm import get_chat_model

# === Step 1. 初始化 Qwen-Agent 模型 ===
llm = get_chat_model(
    {
        "model": "qwen2.5:7b",
        "model_server": "http://localhost:8000/v1",  # 你的本地推論端點
        "api_key": "EMPTY",  # 如果無需驗證，可任意填
        "generate_cfg": {
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": True
                }  # 設定是否開啟 reasoning
            }
        },
    }
)


# === Step 2. 定義可被呼叫的函數 ===
def get_current_temperature(location, unit="celsius"):
    print(f"[CALL] get_current_temperature: {location}, unit={unit}")
    return {"temperature": 26.1, "location": location, "unit": unit}


def get_temperature_date(location, date, unit="celsius"):
    print(f"[CALL] get_temperature_date: {location}, date={date}, unit={unit}")
    return {"temperature": 25.9, "location": location, "date": date, "unit": unit}


def get_function_by_name(name):
    return {
        "get_current_temperature": get_current_temperature,
        "get_temperature_date": get_temperature_date,
    }[name]


# === Step 3. 工具定義 (JSON Schema 格式) ===
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_temperature",
            "description": "Get current temperature at a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location to get the temperature for, in the format 'City, State, Country'.",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The unit to return the temperature in.",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_temperature_date",
            "description": "Get temperature at a location and date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "date": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location", "date"],
            },
        },
    },
]

FUNCTIONS = [tool["function"] for tool in TOOLS]

# === Step 4. 建立對話輸入 ===
MESSAGES = [
    {
        "role": "user",
        "content": "What's the temperature in San Francisco now? How about tomorrow? Current Date: 2024-09-30.",
    }
]

# === Step 5. 模型第一次輸出（生成函數呼叫） ===
for responses in llm.chat(messages=MESSAGES, functions=FUNCTIONS):
    pass
MESSAGES.extend(responses)

# === Step 6. 執行模型要求的函數呼叫 ===
for message in responses:
    if fn_call := message.get("function_call"):
        fn_name = fn_call["name"]
        fn_args = json.loads(fn_call["arguments"])
        fn_res = json.dumps(get_function_by_name(fn_name)(**fn_args))
        MESSAGES.append(
            {
                "role": "function",
                "name": fn_name,
                "content": fn_res,
            }
        )

# === Step 7. 再次呼叫模型，讓它整合函數結果 ===
for final_responses in llm.chat(messages=MESSAGES, functions=FUNCTIONS):
    pass

print("\n=== 💬 最終回答 ===")
for msg in final_responses:
    if msg["role"] == "assistant":
        if "reasoning_content" in msg:
            print(f"[🧠 推理過程]\n{msg['reasoning_content']}\n")
        print(f"[🗣️ 回覆內容]\n{msg['content']}")
