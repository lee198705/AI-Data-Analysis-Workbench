import os

try:
    from openai import OpenAI
except ModuleNotFoundError:
    OpenAI = None


SILICONFLOW_API_KEY_ENV = "SILICONFLOW_API_KEY"
SILICONFLOW_BASE_URL_ENV = "SILICONFLOW_BASE_URL"
SILICONFLOW_MODEL_ENV = "SILICONFLOW_MODEL"
DEFAULT_SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_SILICONFLOW_MODEL = "deepseek-ai/DeepSeek-V4-Flash"


def get_config_value(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value:
        return value

    try:
        import streamlit as st

        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        return default

    return default


def get_siliconflow_config() -> dict[str, str | None]:
    return {
        "api_key": get_config_value(SILICONFLOW_API_KEY_ENV),
        "base_url": get_config_value(SILICONFLOW_BASE_URL_ENV, DEFAULT_SILICONFLOW_BASE_URL),
        "model": get_config_value(SILICONFLOW_MODEL_ENV, DEFAULT_SILICONFLOW_MODEL),
    }


def build_client():
    config = get_siliconflow_config()
    if OpenAI is None:
        raise RuntimeError("缺少 openai 依赖，请先运行：pip install -r requirements.txt")
    if not config["api_key"]:
        raise RuntimeError("未检测到 SILICONFLOW_API_KEY。请先设置硅基流动 API Key 后再使用智能问答。")

    return OpenAI(api_key=config["api_key"], base_url=config["base_url"]), config["model"]


def build_messages(question: str, history: list[dict], max_history: int = 8) -> list[dict]:
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个统计学与数据分析助手。回答要准确、简洁。"
                "如果用户问题依赖参考资料，请优先依据参考资料回答并保留来源编号。"
            ),
        }
    ]

    for message in history[-max_history:]:
        role = message.get("role")
        content = message.get("content")
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": question})
    return messages


def generate_response(question: str, history: list[dict] | None = None) -> str:
    client, model = build_client()
    response = client.chat.completions.create(
        model=model,
        messages=build_messages(question, history or []),
        temperature=0.7,
        top_p=0.8,
        max_tokens=4096,
    )
    return response.choices[0].message.content
