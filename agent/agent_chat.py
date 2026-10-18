# -*- coding: utf-8 -*-
"""githot.agent_chat —— GitHub 热门仓库 Agent：流式输出 + 多轮对话封装。

设计要点（为什么这样封装）
--------------------------------------------------------------------
1. 多轮记忆靠 checkpointer，不靠自己拼 messages 列表。
   create_agent(checkpointer=InMemorySaver()) 之后，同一个 thread_id 的
   对话状态由框架自动累积，每轮只需把【最新一条用户消息】传进去；
   换 thread_id == 重开一局（clear()），进程退出即清空。
   要跨进程持久化，把 InMemorySaver 换成 langgraph.checkpoint.sqlite.SqliteSaver
   （需 pip install langgraph-checkpoint-sqlite），其余代码不用改。

2. 流式输出用 stream_mode=["messages", "values"] 双通道：
   - "messages"：模型逐字增量（AIMessageChunk），用来即时打印，体验好；
   - "values" ：每个节点执行完的完整状态，用来拿工具调用信息、本轮最终答案，
     以及"模型一个字都没流出来"（整轮只有工具调用）时的兜底。
   注意：只有 langgraph_node == "model" 的增量才是给用户看的正文，
   工具节点 / 中间件产生的 chunk 要过滤掉，否则会打印出乱七八糟的东西。

3. 本文件对外只暴露两个东西：
   - AgentChat.send(text)   → 跑一轮，返回最终答案字符串（可选是否边跑边打印）
   - AgentChat.stream(text) → 生成器，产出 AgentEvent 事件流，交给上层自己渲染
     （Web/SSE、富文本终端等都基于 stream() 封装，不用改本文件）

4. 展示层杂活（消息文本归一、工具结果预览、终端逐字打印）不放在本文件，
   统一在 agent/utils.py：message_text / preview / print_event，
   本文件顶部 import 使用，别的会话类也能直接复用。
"""

from __future__ import annotations

import sys
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

# 包内 import 统一走 "githot.xxx"，这样同一进程里每个模块只会被加载一次
# （若混用 agent.xxx 与 githot.agent.xxx，agent_chat 会被加载两遍 → 两个
#  checkpointer → 不同入口之间会话记忆对不上）。
# 但直接 python githot/agent/agent_chat.py 时，"githot" 不在 import 路径上，
# 所以把 githot/ 与项目根都补进 sys.path，并按 __package__ 选择写法，保证四种跑法都可用：
#   python githot/agent/agent_chat.py
#   python -m githot / from githot import AgentChat
#   cd githot && python -m agent.agent_chat
_SELF_DIR = str(Path(__file__).resolve().parent)      # .../githot/agent
_PKG_ROOT = str(Path(_SELF_DIR).parent)               # .../githot
_ROOT_DIR = str(Path(_PKG_ROOT).parent)               # 项目根
for _p in (_ROOT_DIR, _PKG_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if __package__ and __package__.startswith("githot"):
    # 作为 githot 包的子模块被导入（python -m githot / from githot import ... 走这里）
    from githot.agent.llm_model import get_llm
    from githot.agent.utils import message_text, preview, print_event
    from githot.prompt.github_prompt import AGENT_GITHUB_PROMPT
    from githot.tools.github_tools import get_github_rank, get_github_recent_stars_rank
    from githot.tools.wechat_tools import generate_wechat_article
else:
    # 以 githot/ 为根的顶层写法（直接跑文件、或 cd githot 后 -m 跑）
    from agent.llm_model import get_llm                                          # noqa: E402
    from agent.utils import message_text, preview, print_event                   # noqa: E402
    from prompt.github_prompt import AGENT_GITHUB_PROMPT                         # noqa: E402
    from tools.github_tools import get_github_rank, get_github_recent_stars_rank  # noqa: E402
    from tools.wechat_tools import generate_wechat_article                       # noqa: E402

# ===== Agent 构建（模块级单例，多个会话共用同一张图，靠 thread_id 隔离记忆）=====

tools = [get_github_rank, get_github_recent_stars_rank, generate_wechat_article]

model = get_llm()

# 检查点：按 thread_id 存对话状态（含工具调用链）。生产环境换 SqliteSaver/PostgresSaver。
checkpointer = InMemorySaver()

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=AGENT_GITHUB_PROMPT,  # 不传的话提示词根本不生效
    checkpointer=checkpointer,          # 不传的话多轮对话没有记忆
)


# ===== 事件流数据结构 =====

@dataclass
class AgentEvent:
    """一轮对话里的一个流式事件。kind 取值：

    - "token"       模型正文增量，data 是文本片段（逐字，用于流式打印）
    - "thinking"    模型思考过程增量（Qwen enable_thinking），data 是思考文本
    - "tool_call"   模型请求调用工具，name=工具名，data=参数 dict
    - "tool_result" 工具执行完毕，name=工具名，data=工具返回内容（截断后的预览）
    - "final"       本轮最终答案，data=完整答案文本
    """

    kind: str
    data: str = ""
    name: str = ""
    meta: dict = field(default_factory=dict)


class AgentChat:
    """一个会话 = 一个 thread_id。流式输出 + 多轮记忆都封装在这里。

    用法::

        chat = AgentChat()
        chat.send("最近7天 Python 高星新仓库前3")   # 边流式打印边返回完整答案
        chat.send("换成看涨星速度的")               # 自动带上一轮的语言/天数条件
        chat.clear()                               # 重开一局，忘记历史
    """

    # 终端里退出 / 清空的输入词
    EXIT_COMMANDS = {"退出", "exit", "quit", ":q"}
    CLEAR_COMMANDS = {"清除", "清空", "clear", "reset"}
    HISTORY_COMMANDS = {"历史", "history", "/history"}

    def __init__(self, graph=None, thread_id: str | None = None, show_thinking: bool = False):
        """
        Args:
            graph: 自定义 agent 图，默认用模块级的 agent（已挂 checkpointer）。
            thread_id: 指定会话 id；不传则新生成一个。同一 thread_id 可恢复同一份记忆。
            show_thinking: 是否把模型的思考过程（reasoning_content）也输出。
        """
        self.agent = graph if graph is not None else agent
        self.thread_id = thread_id or str(uuid.uuid4())
        self.config = {"configurable": {"thread_id": self.thread_id}}
        self.show_thinking = show_thinking

    # ---------- 核心：流式事件生成器 ----------

    def stream(self, user_input: str) -> Iterator[AgentEvent]:
        """把用户输入跑一轮，产出事件流（不打印任何东西，交给调用方渲染）。

        多轮记忆：有 checkpointer 后只传最新一条 HumanMessage，
        历史由框架按 thread_id 自动拼接，无需自己维护 messages。
        """
        yield from self._stream_input({"messages": [HumanMessage(content=user_input)]})

    def _stream_input(self, input_payload) -> Iterator[AgentEvent]:
        """真正跑 agent.stream 的地方（send 与后续可能的中断续跑都复用它）。"""
        announced_calls: set[str] = set()    # 已播报的"工具调用"id
        announced_results: set[str] = set()  # 已播报的"工具结果"id（与上面分开，否则会互相吞掉）
        emitted_final = False                # 正文是否已经逐字流出来过

        for mode, data in self.agent.stream(
            input_payload,
            config=self.config,
            stream_mode=["messages", "values"],
        ):
            if mode == "messages":
                chunk, metadata = data
                if metadata.get("langgraph_node") != "model":
                    continue
                # 思考过程（Qwen enable_thinking）在 additional_kwargs 里，不是正文
                reasoning = (getattr(chunk, "additional_kwargs", {}) or {}).get("reasoning_content")
                if reasoning and self.show_thinking:
                    yield AgentEvent("thinking", str(reasoning))
                # .text 只取文本部分（str 或 list 内容都能处理），工具调用分片返回空串
                text = getattr(chunk, "text", None)
                if text is None:
                    text = chunk.content if isinstance(chunk.content, str) else ""
                if text:
                    emitted_final = True
                    yield AgentEvent("token", str(text))
                continue

            # values 模式：data 是本轮节点执行后的完整状态
            last = data["messages"][-1] if data.get("messages") else None
            if last is None:
                continue

            if isinstance(last, AIMessage) and last.tool_calls:
                for call in last.tool_calls:
                    if call["id"] in announced_calls:
                        continue
                    announced_calls.add(call["id"])
                    yield AgentEvent("tool_call", str(call.get("args") or {}), name=call["name"],
                                     meta={"tool_call_id": call["id"]})

            elif isinstance(last, ToolMessage):
                if last.tool_call_id in announced_results:
                    continue
                announced_results.add(last.tool_call_id)
                yield AgentEvent("tool_result", preview(last.content), name=str(last.name or ""),
                                 meta={"tool_call_id": last.tool_call_id})

            elif isinstance(last, AIMessage):
                # 本轮最终答案：逐字流式（token 事件）已给用户看过了，
                # 这里再补一个 final 事件，让上层拿到"完整答案"用于返回/记录。
                answer = message_text(last)
                if answer:
                    yield AgentEvent("final", answer, meta={"already_streamed": emitted_final})

    # ---------- 对外：发一条消息，拿完整回答 ----------

    def send(self, user_input: str, echo: bool = True) -> str:
        """发一条消息并返回最终答案。

        Args:
            user_input: 用户这轮说的话。
            echo: True 时在终端逐字流式打印（含工具调用提示）；
                  False 时静默跑完只返回文本（给 Web 接口 / 测试用）。
        """
        answer = ""
        streamed = False
        for event in self.stream(user_input):
            if event.kind == "final":
                answer = event.data
            else:
                if event.kind == "token":
                    streamed = True
                if echo:
                    print_event(event)
        # 兜底：极少数情况下模型没有逐字增量（答案只在完整状态里），补打一次
        if echo and answer and not streamed:
            print(answer)
        if echo:
            print()  # 流式输出后补一个换行，避免和后续提示黏在一行
        return answer

    def ask(self, user_input: str) -> str:
        """send() 的静默版：不打印，直接返回完整答案字符串。"""
        return self.send(user_input, echo=False)

    # ---------- 会话管理 ----------

    def clear(self) -> str:
        """重开一局：换新 thread_id，旧对话状态不再被引用。返回新的 thread_id。"""
        self.thread_id = str(uuid.uuid4())
        self.config = {"configurable": {"thread_id": self.thread_id}}
        return self.thread_id

    def history(self) -> list:
        """当前会话的消息历史（含 Human/AI/Tool），调试用。"""
        snapshot = self.agent.get_state(self.config)
        return list(snapshot.values.get("messages", [])) if snapshot else []

    # ---------- 命令行多轮对话 ----------

    def run_cli(self, banner: bool = True) -> None:
        """终端 REPL：输入问题继续对话，输入"退出"结束，输入"清除"重开一局。"""
        if banner:
            print("=" * 52)
            print("GitHub 热门仓库助手（流式输出 + 多轮记忆）")
            print("输入“退出”结束会话；输入“清除”清空上下文；输入“历史”查看消息数")
            print(f"当前会话 thread_id：{self.thread_id}")
            print("=" * 52)

        while True:
            try:
                user_input = input("\n你：").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n已退出。")
                break

            if not user_input:
                continue
            lowered = user_input.lower()
            if lowered in self.EXIT_COMMANDS or user_input in {"退出"}:
                print("已退出。")
                break
            if lowered in self.CLEAR_COMMANDS or user_input in {"清除", "清空"}:
                self.clear()
                print("已清空对话历史，开始新会话。")
                continue
            if lowered in self.HISTORY_COMMANDS:
                msgs = self.history()
                print(f"当前会话共 {len(msgs)} 条消息（含工具调用记录）。")
                continue

            print("助手：", end="", flush=True)
            self.send(user_input)


if __name__ == "__main__":
    AgentChat().run_cli()
