# -*- coding: utf-8 -*-
"""githot.agent.utils —— 会话封装的配套小工具（文本归一 / 预览 / 终端渲染）。

为什么要单独成文件：agent_chat.py 只该管"怎么跑一轮对话"，
而"消息内容怎么变成干净文本"、"终端怎么打印"属于展示层杂活，
拆出来后可被别处复用（如 rag_demo、weather1 的会话类），也方便单测。

对外三个函数：
    message_text(message|content) → str   把消息/裸 content 归一成纯文本
    preview(content, limit=200)   → str    工具返回值压成一行并截断
    print_event(event)                      AgentEvent 的终端渲染（send(echo=True) 用）
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # 只为类型标注引入；运行时不导入，避免 utils ←→ agent_chat 循环 import。
    from agent.agent_chat import AgentEvent


# ===== 文本归一 =====

def message_text(message: Any) -> str:
    """把"一条消息"或"裸 content"统一转成纯文本字符串。

    LangChain 1.x 里 message.content 不再保证是 str，可能是内容块列表：
        "你好"                                        # 老式：纯字符串
        [{"type": "text", "text": "你好"}, {...}]      # 新式：block 列表
    不转换就会打印出一坨 list 字面量，且 thinking 等块会混进正文。
    所以只挑 type == "text" 的块拼接，其余类型（thinking/tool_use…）忽略。

    传进来的是消息对象就用 getattr 取 .content；是裸 content（str/list）则直接处理。
    """
    content = getattr(message, "content", message)
    return _content_to_text(content)


def _content_to_text(content: Any) -> str:
    """content 的多形态归一（str / block 列表 / 其他对象兜底）。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


def preview(content: Any, limit: int = 200) -> str:
    """把工具返回内容压成一行、截断到 limit 字，避免终端被大段 JSON 刷屏。

    完整结果仍留在 checkpointer 的消息里给模型看，这里只是"给人看的摘要"。
    """
    text = " ".join(_content_to_text(content).split())   # 折掉换行与多余空格
    return text if len(text) <= limit else text[:limit] + "…"


# ===== 终端渲染 =====

def print_event(event: AgentEvent) -> None:
    """把一个 AgentEvent 打到终端：正文逐字流式、工具提示单独占行。

    这是"事件流"与"如何显示"的分界：stream() 只产事件不打印，
    换成 Web/SSE 时只需另写一个渲染函数，会话逻辑一行不用改。
    """
    if event.kind == "token":
        print(event.data, end="", flush=True)          # 逐字追加、立刻可见 = 流式观感
    elif event.kind == "thinking":
        print(f"\n[思考] {event.data}", end="", flush=True)
    elif event.kind == "tool_call":
        print(f"\n[调用工具 {event.name}] {event.data}", flush=True)
    elif event.kind == "tool_result":
        print(f"[工具 {event.name} 返回] {event.data}\n[正在整理回答…]", flush=True)
    # final 不打印：正文已由 token 逐字输出过，重复打会刷屏
