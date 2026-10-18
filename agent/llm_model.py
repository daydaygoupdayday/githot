import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from functools import lru_cache

load_dotenv()

@lru_cache(maxsize=None)
def get_llm(
        model: str | None = None,
        temperature: float | None = None,
        enable_thinking: bool = True,        
    ) -> ChatOpenAI:
    """创建（并缓存）ChatOpenAI 实例。
        Args:
            model: 模型名；不传则使用 .env 中 MODEL 的配置。
            temperature: 采样温度；None 表示不传、使用服务端默认值。
            enable_thinking: Qwen 思考模式开关。注意：结构化输出的
                ToolStrategy 会强制 tool_choice，与思考模式冲突，
                那种场景要传 enable_thinking=False。
        Returns:
            ChatOpenAI 实例。相同参数永远返回同一个对象（lru_cache 缓存）。
    """
    args = {}

    if temperature is not None:
        args["temperature"] = temperature
    if not enable_thinking:
        args["extra_body"] = {"enable_thinking": False}

    return ChatOpenAI(
        api_key=os.getenv("API_KEY"), 
        base_url=os.getenv("BASE_URL"),
        model=model or os.getenv("MODEL"),
        **args,
    )

def reset_llm_cache() -> None:
    """清空所有缓存的模型实例。
    修改 .env 后想热重载配置、或在测试里隔离用例时调用。
    下次 get_llm() 会用新的环境变量重新创建实例。
    """
    get_llm.cache_clear()