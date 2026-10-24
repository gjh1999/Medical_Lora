"""Moonshot API connectivity smoke test.

Set MOONSHOT_API_KEY before running this script. Never commit API keys.
"""

import os

from openai import OpenAI


api_key = os.environ.get("MOONSHOT_API_KEY")
if not api_key:
    raise RuntimeError("请先设置 MOONSHOT_API_KEY 环境变量")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.moonshot.cn/v1",
    timeout=30,
)

response = client.chat.completions.create(
    model="kimi-k2.6",
    messages=[
        {"role": "system", "content": "你是一个助手。"},
        {"role": "user", "content": "你好，只回复OK"},
    ],
    max_tokens=50,
)

print(response.choices[0].message.content)
