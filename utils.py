import requests
import json


def call_model():
    api_key = "sk-rHa4hQLoPiQpGl7imxu6lp1nLjxtdyxcGf3j1afaCOKV1grE"
    url = "https://api.aipaibox.com/v1/chat/completions"
    prompt = ''
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    data = {"model": "gpt-5.5",
            "messages": [{"role": "user", "content": prompt}],
            # "response_format": {"type": "json_object"},
            "stream": False}
    response = requests.post(url, headers=headers, json=data)
    result = response.json()["choices"][0]["message"]["content"]
    return result

def call_deepseek(prompt, timeout=600):
    url = "https://api.deepseek.com/v1/responses"
    headers = {"Authorization": f"Bearer sk-7802ae9cf2764f09ad074fe23bcbe74c", "Content-Type": "application/json"}
    data = {"model": "deepseek-v4-flash", "input": prompt, "response_format": {"type": "json_object"}}
    data["tools"] = [{"type": "web_search"}]
    data["tool_choice"] = "auto"

    response = requests.post(url, headers=headers, json=data, timeout=timeout)
    response.raise_for_status()
    result = response.json()
    output_text = []
    for item in result.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                text = content.get("text")
                if text:
                    output_text.append(text)
    return "".join(output_text)


def call_deepseek_stream(prompt, timeout=600):
    """Call DeepSeek API with streaming support"""
    url = "https://api.deepseek.com/v1/responses"
    headers = {"Authorization": f"Bearer sk-7802ae9cf2764f09ad074fe23bcbe74c", "Content-Type": "application/json"}
    data = {"model": "deepseek-v4-flash", "input": prompt, "stream": True}
    data["tools"] = [{"type": "web_search"}]
    data["tool_choice"] = "auto"

    response = requests.post(url, headers=headers, json=data, timeout=timeout, stream=True)
    response.raise_for_status()

    for line in response.iter_lines():
        if line:
            line_str = line.decode('utf-8')
            if line_str.startswith('data: '):
                json_str = line_str[6:]
                if json_str.strip() == '[DONE]':
                    break
                try:
                    chunk = json.loads(json_str)
                    yield chunk
                except json.JSONDecodeError:
                    continue

