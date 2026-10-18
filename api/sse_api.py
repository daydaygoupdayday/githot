# -*- coding: utf-8 -*-
"""githot.api.sse_api —— 把 githot 功能封装成 HTTP/SSE 接口，供网页端调用。

设计要点
--------------------------------------------------------------------
1. 不重复造轮子：对话直接复用 AgentChat.stream()（事件流原样转发成 SSE），
   文章直接复用 wechat_tools 的三层管线（fetch_rank_records / stream_article /
   generate_article）。本文件只做"协议适配"：把 Python 生成器变成 SSE 事件流。

2. SSE 而不是 WebSocket：浏览器端 EventSource 原生支持、断线自动重连、
   经过代理/网关也稳定；逐字输出用 POST+SSE（fetch 读流）与 GET+SSE（EventSource）
   双形式都提供，前端按需选。

3. 每个 HTTP 请求 = 一次"跑生成器"的循环，agent.stream() 是同步阻塞生成器，
   放到 FastAPI 的线程池执行（def 路由自动进线程池；流式响应经
   run_in_threadpool 包装），不堵事件循环。

4. 错误语义：
   - 流还没开始（参数校验失败）→ 正常 HTTP 状态码 4xx/5xx + JSON；
   - 流已经开始（HTTP 头已发出，状态码改不了）→ 只发一个 error 事件再结束，
     前端统一在 onerror/error 事件里处理。

5. 鉴权：环境变量 GITHOT_API_TOKEN 设置后，除 /health 外所有接口要求
   Authorization: Bearer <token>；不设置则完全开放（本机开发默认）。

启动：
    python -m githot.api.sse_api            # 仓库根（githot/ 的上级）或包内均可
    uvicorn githot.api.sse_api:app --port 8000
    # 在 githot/ 目录内直接跑文件：
    python api/sse_api.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Iterator

# 与 agent_chat 同样的"两种写法"策略：保证包内导入与直接跑文件都可用
_PKG_ROOT = str(Path(__file__).resolve().parent.parent)      # .../githot
_ROOT_DIR = str(Path(_PKG_ROOT).parent)                       # 项目根
for _p in (_ROOT_DIR, _PKG_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if __package__ and __package__.startswith("githot"):
    from githot.agent.agent_chat import AgentChat
    from githot.prompt.wechat_prompt import list_article_styles
    from githot.tools.wechat_tools import (
        BOARD_DEFAULT_MIN_STARS,
        BOARD_LABEL,
        fetch_rank_records,
        stream_article,
    )
else:
    from agent.agent_chat import AgentChat                                  # noqa: E402
    from prompt.wechat_prompt import list_article_styles                    # noqa: E402
    from tools.wechat_tools import (                                         # noqa: E402
        BOARD_DEFAULT_MIN_STARS,
        BOARD_LABEL,
        fetch_rank_records,
        stream_article,
    )

from fastapi import FastAPI, HTTPException, Query, Request                  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware                          # noqa: E402
from fastapi.responses import JSONResponse, StreamingResponse               # noqa: E402
from pydantic import BaseModel, Field, field_validator                      # noqa: E402

# ===== 应用与配置 =====

app = FastAPI(
    title="githot API",
    description="GitHub 热榜 Agent 的 HTTP/SSE 接口：多轮对话、榜单查询、公众号文章生成。",
    version="1.0.0",
)

ALLOW_ORIGINS = [o.strip() for o in os.getenv("GITHOT_ALLOW_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def bearer_auth(request: Request, call_next):
    """GITHOT_API_TOKEN 设置后启用 Bearer 鉴权（/health 与 OPTIONS 预检放行）。"""
    token = os.getenv("GITHOT_API_TOKEN", "").strip()
    path = request.url.path
    if token and path not in {"/health", "/docs", "/openapi.json"} and request.method != "OPTIONS":
        auth = request.headers.get("Authorization", "")
        if auth != f"Bearer {token}":
            return JSONResponse(status_code=401, content={"detail": "无效或缺失的 Authorization 头"})
    return await call_next(request)


# ===== SSE 编码工具 =====

def _sse(event: str, data) -> str:
    """把一条事件编码成 SSE 帧；data 一律 JSON 化，前端 JSON.parse 即可。"""
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def _stream_response(events: Iterator[str]) -> StreamingResponse:
    return StreamingResponse(
        events,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# ===== 参数模型：把模型/前端可能传来的脏值在入口统一洗成 int =====

def _clean_int(value, default: int, lo: int, hi: int, name: str) -> int:
    if value is None:
        return default
    try:
        n = int(float(str(value).strip()))
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail=f"{name} 必须是数字，收到：{value!r}")
    return max(lo, min(n, hi))


def _clean_opt_str(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text and text.lower() not in {"none", "null", "nil"} else None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    thread_id: str = Field(default="default", max_length=128)
    show_thinking: bool = False


class ArticleRequest(BaseModel):
    board: str = Field(default="hot")
    days_age: int | str | None = 7
    language: str | None = None
    min_stars: int | str | None = None
    rank: int | str | None = 5
    style: str | None = None
    word_count: int | str | None = None

    @field_validator("board")
    @classmethod
    def _board_known(cls, v: str) -> str:
        v = (v or "hot").strip().lower()
        if v not in BOARD_LABEL:
            raise ValueError(f"board 只支持 {list(BOARD_LABEL)}，收到：{v!r}")
        return v


def _article_params(req: ArticleRequest):
    """校验 + 规范化文章参数，返回 (board, days_age, language, min_stars, rank, style, word_count)。"""
    board = (req.board or "hot").strip().lower()
    if board not in BOARD_LABEL:
        raise HTTPException(status_code=422, detail=f"board 只支持 {list(BOARD_LABEL)}，收到：{board!r}")
    days_age = _clean_int(req.days_age, 7, 1, 365, "days_age")
    rank = _clean_int(req.rank, 5, 1, 30, "rank")
    min_stars = _clean_int(req.min_stars, BOARD_DEFAULT_MIN_STARS[board], 0, 1_000_000, "min_stars")
    language = _clean_opt_str(req.language)
    style = _clean_opt_str(req.style)
    word_count = _clean_int(req.word_count, 800, 200, 5000, "word_count") if req.word_count not in (None, "") else None
    return board, days_age, language, min_stars, rank, style, word_count


# ===== 基础接口 =====

@app.get("/health")
def health():
    return {"status": "ok", "service": "githot-api"}


@app.get("/api/article/styles")
def article_styles():
    return {"styles": list_article_styles(), "default": "科技种草"}


@app.get("/api/boards")
def boards():
    return {"boards": [{"value": k, "label": v} for k, v in BOARD_LABEL.items()],
            "default_min_stars": BOARD_DEFAULT_MIN_STARS}


# ===== 对话接口 =====

def _chat_events(req: ChatRequest) -> Iterator[str]:
    """跑一轮 AgentChat.stream()，把 AgentEvent 转成 SSE 事件。"""
    chat = AgentChat(thread_id=req.thread_id, show_thinking=req.show_thinking)
    yield _sse("meta", {"thread_id": chat.thread_id})
    try:
        for ev in chat.stream(req.message):
            if ev.kind == "token":
                yield _sse("token", {"text": ev.data})
            elif ev.kind == "thinking":
                yield _sse("thinking", {"text": ev.data})
            elif ev.kind == "tool_call":
                yield _sse("tool_call", {"name": ev.name, "args": ev.data})
            elif ev.kind == "tool_result":
                yield _sse("tool_result", {"name": ev.name, "preview": ev.data})
            elif ev.kind == "final":
                yield _sse("final", {"text": ev.data})
        yield _sse("done", {})
    except Exception as e:  # 流已开始，只能以 error 事件收尾
        yield _sse("error", {"detail": f"{type(e).__name__}: {e}"})


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest):
    return _stream_response(_chat_events(req))


@app.get("/api/chat/stream")
def chat_stream_get(message: str = Query(min_length=1), thread_id: str = "default",
                    show_thinking: bool = False):
    """GET 形式：浏览器 EventSource 直接可用（POST 版供 axios/fetch 用）。"""
    return _stream_response(_chat_events(ChatRequest(message=message, thread_id=thread_id,
                                                     show_thinking=show_thinking)))


# ===== 文章接口 =====

def _article_stream_events(req: ArticleRequest) -> Iterator[str]:
    board, days_age, language, min_stars, rank, style, word_count = _article_params(req)
    meta, records, error = fetch_rank_records(board, days_age, language, min_stars, rank)
    if error:
        yield _sse("error", {"detail": error})
        return
    if not records:
        yield _sse("error", {"detail": "没有符合条件的新仓库，请放宽条件（增大天数或降低星数门槛）"})
        return
    # meta 事件先给文章口径与真实条目：前端把仓库名/链接/星数以这里为准展示，
    # 不依赖模型输出（模型只负责行文）。
    yield _sse("meta", {"meta": meta, "items": records})
    try:
        for text in stream_article(records, meta, style=style, word_count=word_count):
            yield _sse("token", {"text": text})
        yield _sse("done", {})
    except Exception as e:
        yield _sse("error", {"detail": f"生成文章失败 {type(e).__name__}: {e}"})


@app.post("/api/article/stream")
def article_stream(req: ArticleRequest):
    return _stream_response(_article_stream_events(req))


@app.get("/api/article/stream")
def article_stream_get(board: str = "hot", days_age: int = 7, language: str | None = None,
                       min_stars: int | None = None, rank: int = 5,
                       style: str | None = None, word_count: int | None = None):
    req = ArticleRequest(board=board, days_age=days_age, language=language, min_stars=min_stars,
                         rank=rank, style=style, word_count=word_count)
    return _stream_response(_article_stream_events(req))


@app.post("/api/article")
def article_once(req: ArticleRequest):
    """阻塞式一次性返回：适合脚本/测试；网页流式体验用 /api/article/stream。"""
    board, days_age, language, min_stars, rank, style, word_count = _article_params(req)
    meta, records, error = fetch_rank_records(board, days_age, language, min_stars, rank)
    if error:
        raise HTTPException(status_code=502, detail=error)
    if not records:
        raise HTTPException(status_code=404, detail="没有符合条件的新仓库，请放宽筛选条件")
    try:
        article = stream_and_join(records, meta, style, word_count)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"生成文章失败：{type(e).__name__}: {e}")
    return {"meta": meta, "items": records, "article": article}


def stream_and_join(records, meta, style, word_count) -> str:
    return "".join(stream_article(records, meta, style=style, word_count=word_count))


# ===== 会话管理接口 =====

@app.get("/api/threads/{thread_id}/messages")
def thread_messages(thread_id: str):
    """查看某会话的消息历史：角色 / 长度 / 预览（不含工具原始大 JSON）。"""
    chat = AgentChat(thread_id=thread_id)
    msgs = chat.history()
    out = []
    for m in msgs:
        cls = type(m).__name__
        role = {"HumanMessage": "user", "AIMessage": "assistant", "ToolMessage": "tool"}.get(cls, cls)
        text = ""
        content = getattr(m, "content", "")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
        out.append({"role": role, "name": getattr(m, "name", None),
                    "length": len(text), "preview": text[:120]})
    return {"thread_id": thread_id, "count": len(out), "messages": out}


@app.delete("/api/threads/{thread_id}")
def clear_thread(thread_id: str):
    """清空会话：in-memory checkpointer 按 thread_id 隔离，换新 id 即等效清空。"""
    return {"thread_id": thread_id, "cleared": True,
            "note": "前端换一个 thread_id 即可开始新会话"}


# ===== 直接运行 =====

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("GITHOT_HOST", "0.0.0.0")
    port = int(os.getenv("GITHOT_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
