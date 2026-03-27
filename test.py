from openai import OpenAI
import yaml
import os
import requests
import json

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

LM_STUDIO_HOST = config["lm_studio"]["host"]
LM_STUDIO_PORT = config["lm_studio"]["port"]
LM_STUDIO_KEY = config["lm_studio"]["api_key"]
DEFAULT_MODEL = config["app"]["default_model"]

HISTORY_FILE = config["app"]["history_file_name"]
MAX_HISTORY_ITEMS = config["app"]["history_lemgth"]

client = OpenAI(base_url=f"http://{LM_STUDIO_HOST}:{LM_STUDIO_PORT}/v1", api_key=LM_STUDIO_KEY)


def schema_response(model:str,prompt:str,text:str):
    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "system", "content": prompt},{"role": "user", "content": text}],
                        temperature=0.0,  # Низкая температура для консистентности
                        response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "score",
                            "schema": {
                        "properties": {
                            "readability": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 100
                            },
                            "quality": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 100
                            },
                            "creativity": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 100
                            }
                        },
                        "required": [
                            "readability",
                            "quality",
                            "creativity"
                        ]
                        },
                            }
                        }
                                        )
    return response.choices[0].message.content

def thinking_response(model:str,prompt:str,text:str):
    system_content = """<|start_header_id|>system<|end_header_id|>
Reasoning_effort: high
Tools: none
<|eot_id|>"""
    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "system", "content": system_content},{"role": "user", "content": text+prompt}],
                        temperature=0.0,  # Низкая температура для консистентности
                        reasoning_effort="high",
                        extra_body={"reasoning_level": "high"}
                        )
    return response.choices[0].message.content

api_token = os.environ.get('LM_API_TOKEN', 'your_token_here') 

response = requests.post(
    "http://localhost:1234/api/v1/chat",
    headers={
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    },
    json={
        "model": "openai/gpt-oss-20b",  # Укажите точное имя вашей модели
        "input": "Решите сложную логическую задачу: ...",
        # ВОТ ЭТОТ ПАРАМЕТР РАБОТАЕТ НАПРЯМУЮ ЧЕРЕЗ НАТИВНЫЙ API
        "reasoning": "high", 
        # Дополнительные опции (опционально)
        "temperature": 0.0
    }
)

if response.status_code == 200:
    result = response.json()
    print(json.dumps(result["output"][1]["content"],indent=2,ensure_ascii=False))
else:
    print(f"Ошибка: {response.status_code}")
    print(response.text)