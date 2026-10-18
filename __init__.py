# -*- coding: utf-8 -*-
"""githot —— GitHub 热门仓库助手包入口。

真正的实现都在 githot.agent.agent_chat（会话/流式）与 githot.agent.utils（展示层小工具），
本文件只做导出：包入口不构建 agent，避免和 agent_chat 重复建图。

命令行多轮对话：
    python -m githot                       # 走本包的 __main__.py
    python githot/agent/agent_chat.py      # 直接跑文件也行
"""

from githot.agent.agent_chat import (
    AgentChat,
    AgentEvent,
    agent,
    checkpointer,
    model,
    tools,
)
from githot.agent.utils import message_text, preview, print_event

__all__ = [
    "AgentChat",
    "AgentEvent",
    "agent",
    "checkpointer",
    "model",
    "tools",
    "message_text",
    "preview",
    "print_event",
]
