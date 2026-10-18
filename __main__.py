# -*- coding: utf-8 -*-
"""python -m githot —— 直接开一个流式多轮对话的终端会话。"""

from agent.agent_chat import AgentChat

if __name__ == "__main__":
    AgentChat().run_cli()
