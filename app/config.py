import yaml
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent

load_dotenv(PROJECT_ROOT / ".env")

with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# 从环境变量覆盖敏感配置
config["rag"]["llm_api_key"] = os.environ.get("DASHSCOPE_API_KEY", config["rag"].get("llm_api_key", ""))
config["elasticsearch"]["host"] = os.environ.get("ES_HOST", config["elasticsearch"]["host"])


def resolve_path(relative: str) -> str:
    """将配置中的相对路径解析为绝对路径（相对于项目根目录）"""
    p = Path(relative)
    if p.is_absolute():
        return str(p)
    return str(PROJECT_ROOT / relative)
