# -*- coding: utf-8 -*-
"""wechat_tools.py —— 把 GitHub 排行榜数据写成公众号文章（工具 + 可复用函数）。

为什么工具要"自己取数据"，而不是让模型把刚查到的榜单原样传进来？
--------------------------------------------------------------------
模型转述长 JSON 会漏字段、改星标数、把 description 意译成"事实"，
正好踩中 wechat_prompt 里【数据纪律】要防的那条。所以本工具内部直接复用
github_tools 的检索函数取数 → 拼数据区 → 交给写作模型，
数据全程不经过对话模型的"手抄"，只经过一次程序化 json.dumps。

三层结构（@tool 只是最外面那层薄壳；想用程序直接生成就调里面两层，跳过对话模型）：
    fetch_rank_records()      取数：调 GitHub API，返回 (meta, records, error)
    build_article_messages()  组装：数据区 JSON + 风格提示 → LangChain messages
    write_article()/stream_article()  生成：调写作 LLM，阻塞版 / 流式版
"""

from __future__ import annotations

import datetime
import json
import sys
from collections.abc import Iterator
from pathlib import Path

from langchain.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

# 与 agent_chat 同样的"两种写法"策略：作为 githot 包的子模块加载时用 githot.xxx，
# 直接跑/以 githot 为根加载时用 tools.xxx —— 避免同一模块被加载两份、@tool 对象变两个。
_PKG_ROOT = str(Path(__file__).resolve().parent.parent)      # .../githot
_ROOT_DIR = str(Path(_PKG_ROOT).parent)                       # 项目根
for _p in (_ROOT_DIR, _PKG_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if __package__ and __package__.startswith("githot"):
    from githot.agent.llm_model import get_llm
    from githot.prompt.wechat_prompt import (
        DATA_ZONE_RULES,
        DEFAULT_ARTICLE_STYLE,
        WECHAT_ARTICLE_PROMPT,
        build_layout_seed,
        build_style_block,
    )
    from githot.tools.github_tools import fetch_hot_records, fetch_rising_records
else:
    from agent.llm_model import get_llm                                    # noqa: E402
    from prompt.wechat_prompt import (                                     # noqa: E402
        DATA_ZONE_RULES,
        DEFAULT_ARTICLE_STYLE,
        WECHAT_ARTICLE_PROMPT,
        build_layout_seed,
        build_style_block,
    )
    from tools.github_tools import fetch_hot_records, fetch_rising_records  # noqa: E402


# 两种榜单的默认星标门槛，与 github_tools 里两个查询工具保持一致
BOARD_DEFAULT_MIN_STARS = {"hot": 50, "rising": 10}
BOARD_LABEL = {"hot": "星标总量榜", "rising": "平均每天涨星速度榜"}


def _as_int(value, default: int | None = None) -> int | None:
    """把模型可能传来的字符串数字转成 int，转不了就用默认值。

    为什么必须做：模型抽参经常把整数写成字符串（word_count="800"），
    甚至把 None 写成字符串 "None"/"null"/""。这些值传进 pydantic 校验或
    比较运算会直接报错，整轮工具调用失败、模型再重试一次 → 白等十几秒。
    实测确实出现过 min_stars='None'、word_count='800' 的情况。
    """
    if value is None:
        return default
    if isinstance(value, bool):           # bool 是 int 子类，但显然不是想要的意思
        return default
    if isinstance(value, int):
        return value
    text = str(value).strip().replace(",", "").replace("，", "")
    if text.lower() in {"", "none", "null", "nil", "无", "不填", "默认"}:
        return default
    try:
        return int(float(text))           # "800" / "800.0" / " 800 "
    except ValueError:
        return default


def _as_opt_int(value) -> int | None:
    """可选整数：转不出来就返回 None（让下游用榜单默认值兜底）。"""
    return _as_int(value, None)


# 写作模型：温度稍高更顺；关思考模式换低延迟（文章主要是"按数据改写"，不需要长推理）
WRITER_TEMPERATURE = 0.7
WRITER_ENABLE_THINKING = False


# ===== 第 1 层：取数 =====

def fetch_rank_records(board: str = "hot", days_age: int = 7, language: str | None = None,
                       min_stars: int | None = None, rank: int = 5):
    """按榜单类型取最近新创建仓库，返回 (meta, records, error)。

    成功：error 为 None，records 是仓库记录列表（字段与 github_tools 完全一致，
          涨星榜额外含 stars_per_day / created_at）。
    失败：records 为 None，error 是给用户看的提示串。

    数字参数一律先过 _as_int 兜底 —— 本函数既被程序直接调用（数字已校验），
    也被 @tool 调用（模型可能传字符串），不能假设上游干净。
    """
    board = str(board or "hot").strip().lower()
    if board not in BOARD_LABEL:
        return None, None, f"未知的榜单类型 board={board!r}，只支持 hot（星标总量榜）或 rising（涨星速度榜）。"

    days_age = _as_int(days_age, 7)
    rank = _as_int(rank, 5)
    min_stars = _as_int(min_stars, None)
    if min_stars is None:
        min_stars = BOARD_DEFAULT_MIN_STARS[board]
    language = str(language).strip() if language and str(language).strip().lower() not in {"none", "null", "不限", "所有语言"} else None

    try:
        # 排名逻辑完全复用 github_tools 的公开取数函数，保证"文章里的数据"
        # 和"对话里查到的数据"口径一致（同一份代码，不会出现两边排名不同）
        # detail=True：文章需要 topics/license/forks/created_at 这些具体事实。
        # 事实越具体，句子越有内容，越不需要靠套话凑字数 —— 这是"去机器味"的地基。
        fetcher = fetch_hot_records if board == "hot" else fetch_rising_records
        records, error = fetcher(days_age, language, min_stars, rank, detail=True)
        if error:
            return None, None, error

        meta = {
            "board": board,
            "board_label": BOARD_LABEL[board],
            "days_age": int(days_age),
            "language": language or None,
            "min_stars": int(min_stars),
            "rank": max(1, min(int(rank), 30)),
            "count": len(records),
            "generated_at": datetime.date.today().isoformat(),
        }
        return meta, records, None

    except Exception as e:  # 参数类型错、网络异常等，统一转成人话
        return None, None, f"获取{BOARD_LABEL[board]}数据时出错：{str(e)}"


# ===== 第 2 层：组装提示 =====

def describe_scope(meta: dict) -> str:
    """把筛选口径说成一句自然中文，且同一口径有多种说法。

    为什么要随机：口径串是文章导语唯一的"硬信息"，如果程序里写死一种说法，
    模型就会照着复读，等于给每篇文章盖同一个开头。这里换着说，模型复读哪句都不固定。
    """
    import random

    n = meta["count"]
    lang = meta["language"]
    days, stars = meta["days_age"], meta["min_stars"]
    board = "按星标总量排" if meta["board"] == "hot" else "按平均每天涨星速度排"

    where = f"{lang} 方向" if lang else "各语言混合"
    phrasings = [
        f"只看最近 {days} 天新建的 {where}仓库，门槛是 {stars} 星，{board}，拿到 {n} 个",
        f"样本是新建不满 {days} 天的 {where}仓库，星标至少 {stars}，{board}，共 {n} 条",
        f"{where}，近 {days} 天内创建、{stars} 星起，{board}，{n} 个",
    ]
    return random.choice(phrasings)


def build_article_messages(records: list, meta: dict, style: str | None = None,
                           word_count: int | None = None, extra_requirement: str | None = None,
                           seed: int | None = None) -> list[BaseMessage]:
    """把数据区 + 风格要求 + 本版排版拼成写作模型的 messages。

    数据区用 json.dumps 原样给出（ensure_ascii=False 保留中文），
    写作模型只被允许"照这个写"，所以这里绝不能提前改写或摘要 records。

    seed 控制"本版排版"的随机抽取：默认 None（每次不同）。同一篇想复现就传同一个数。
    """
    system_parts = [
        WECHAT_ARTICLE_PROMPT,
        "## 本篇风格\n" + build_style_block(style),
        build_layout_seed(seed),
        "## 数据区约定\n" + DATA_ZONE_RULES,
    ]

    data_zone = json.dumps({"board_type": meta["board"], "board_label": meta["board_label"],
                            "params": {k: meta[k] for k in ("days_age", "language", "min_stars", "rank", "generated_at")},
                            "items": records}, ensure_ascii=False, indent=2)

    human_parts = [
        "请把下面【数据区】写成一篇微信公众号推文，直接从标题开始输出，不要任何前置说明。",
        # 口径只作为"背景"给一次，并且明确要求别照抄 —— 早先这里是一行
        # "统计口径：近7天…语言=…最低星标=…共N条"，模型会原样搬进导语，
        # 于是每篇开头都是"本次榜单筛选了近X天创建、星标数超过Y的Z仓库，共收录N个项目"，
        # 这句式本身就是指纹。
        f"背景（供你理解数据怎么来的，不要逐项复述成一句话）：{describe_scope(meta)}。",
    ]
    if word_count:
        human_parts.append(f"全文大概 {word_count} 字，允许上下浮动 20%。")
    if extra_requirement:
        human_parts.append(f"用户额外要求（在不违反数据纪律的前提下满足）：{extra_requirement}")
    human_parts.append("【数据区】\n```json\n" + data_zone + "\n```")

    return [SystemMessage(content="\n\n".join(system_parts)), HumanMessage(content="\n\n".join(human_parts))]


# ===== 第 3 层：生成 =====

def write_article(records: list, meta: dict, style: str | None = None, word_count: int | None = None,
                  extra_requirement: str | None = None, seed: int | None = None) -> str:
    """阻塞式生成整篇文章（对话工具用这个，一次返回全文）。"""
    messages = build_article_messages(records, meta, style, word_count, extra_requirement, seed)
    return _writer().invoke(messages).content


def stream_article(records: list, meta: dict, style: str | None = None, word_count: int | None = None,
                   extra_requirement: str | None = None, seed: int | None = None) -> Iterator[str]:
    """流式生成文章，逐段产出文本片段（要边生成边展示就用这个，不用等整篇写完）。"""
    messages = build_article_messages(records, meta, style, word_count, extra_requirement, seed)
    for chunk in _writer().stream(messages):
        text = getattr(chunk, "text", None)
        if text is None:
            text = chunk.content if isinstance(chunk.content, str) else ""
        if text:
            yield text


def _writer():
    return get_llm(temperature=WRITER_TEMPERATURE, enable_thinking=WRITER_ENABLE_THINKING)


def generate_article(board: str = "hot", days_age: int = 7, language: str | None = None,
                     min_stars: int | None = None, rank: int = 5, style: str | None = None,
                     word_count: int | None = None, extra_requirement: str | None = None):
    """一步到位：取数 + 写文章。返回 (meta, records, article, error)，失败时前三项可能为 None。"""
    meta, records, error = fetch_rank_records(board, days_age, language, min_stars, rank)
    if error:
        return None, None, None, error
    if not records:
        # 提示里用 meta 的规范化值（fetch 已把 "7"→7、"None"→默认值处理过）
        return meta, records, None, (
            f"近{meta['days_age']}天（语言={meta['language'] or '不限'}、最低{meta['min_stars']}星）"
            "没有符合条件的新仓库，建议放宽条件：增大 days_age 或降低 min_stars。"
        )
    try:
        article = write_article(records, meta, style, _as_opt_int(word_count), extra_requirement)
    except Exception as e:
        return meta, records, None, f"生成公众号文章失败：{str(e)}"
    return meta, records, article, None


# ===== 暴露给对话 agent 的工具 =====

# 数字参数故意写成 int | str：模型抽参时会传 word_count="800"、min_stars="None"
# 这类字符串（实测出现过）。若 schema 只允许 integer，pydantic 会在校验阶段就拒绝，
# 整轮工具调用失败、模型重试一次 → 白等十几秒。放宽成联合类型后，值会进到函数体，
# 由 _as_int 统一兜底成 int 或默认值，一次成功。
@tool
def generate_wechat_article(board: str = "hot", days_age: int | str = 7, language: str | None = None,
                            min_stars: int | str | None = None, rank: int | str = 5,
                            style: str = DEFAULT_ARTICLE_STYLE, word_count: int | str | None = None):
    """把 GitHub 热门仓库数据写成一篇可直接发布的微信公众号文章。

    当用户说"写成公众号文章/推文/种草文""帮我生成一篇介绍这些仓库的文章""要能直接发的稿子"
    等写作类需求时调用本工具。它会自己查询榜单数据并生成文章，【不需要】你先把榜单结果
    复述给它——你只负责把用户要的筛选条件传对。

    Args:
        board: 榜单类型。"hot"=星标总量榜（默认，取最近新仓库里星标总数最高的）；
            "rising"=涨星速度榜（取平均每天新增星标最快的）。用户说"涨星最快/上升最快"填 rising。
        days_age: 统计最近多少天内新创建的仓库，整数，默认 7。"本周"填 7，"最近一个月"填 30。
        language: 仓库语言过滤，例如 "Python"（首字母大写）。不限语言就【省略该参数】，
            不要填 "None"、"null" 或空字符串。
        min_stars: 仓库最少星标数，整数（如 50）。不限制就【省略该参数】，
            工具会用默认值（hot 为 50，rising 为 10）；不要填字符串 "None"。
        rank: 取前 rank 个仓库写进文章，整数，默认 5，上限 30。用户说"挑10个"就填 10。
        style: 文章风格，可选 "科技种草"（默认）/"热点资讯"/"轻松段子"/"深度解读"。
        word_count: 全文字数目标，整数（如 800）。用户没提就【省略该参数】，
            文章规范会自动控制在 800-1500 字。
    Returns:
        Markdown 格式的完整文章（标题 + 导语 + 逐条仓库小节 + 结尾），
        或一段失败原因（数据为空 / API 限流 / 生成异常）。
    """
    meta, _records, article, error = generate_article(
        board=board, days_age=days_age, language=language,
        min_stars=min_stars, rank=rank, style=style, word_count=word_count,
    )
    if error:
        return error
    return article
