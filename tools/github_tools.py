

from langchain.tools import tool
import requests
import datetime
from rich import print as rprint


def _build_recent_query(days_age, min_stars, language):
    """规范化参数并拼出 GitHub Search 的 q 串，返回 (days_age, min_stars, query)。

    【容错】模型抽参时可能把数字传成字符串（如 days_age="5"），
    timedelta(days="5") 会抛异常，这里强制 int() 兜底。
    【修复】q 里只放 GitHub 支持的限定符（created/stars/language）；
    rank 之类不是限定符，拼进去会让 GitHub 静默返回 200 + 空列表。
    """
    days_age = int(days_age)
    min_stars = int(min_stars)

    date_limit = (datetime.datetime.now() - datetime.timedelta(days=days_age)).strftime("%Y-%m-%d")
    query = f"created:>{date_limit} stars:>={min_stars}"

    if language:
        query += f" language:{language}"

    return days_age, min_stars, query


def _search_new_repos(query, per_page):
    """调用 GitHub Search API 拉取最近新创建仓库，返回 (items, error)。

    成功时 error 为 None、items 为仓库原始 dict 列表；
    失败（限流 / 非 200）时 items 为 None、error 为给用户看的提示串。
    sort/order/per_page 是 URL 查询参数而非 q 限定符；单页上限 30。
    """
    url = "https://api.github.com/search/repositories"
    headers = {"Accept": "application/vnd.github.v3+json", "User-Agent": "Mozilla/5.0"}
    params = {"q": query, "sort": "stars", "order": "desc", "per_page": per_page}
    response = requests.get(url, params=params, timeout=40, headers=headers)

    if response.status_code == 200:
        return response.json()["items"], None
    elif response.status_code == 403:
        return None, "GitHub API 请求过于频繁，请稍后再试。"
    else:
        return None, f"GitHub API 请求失败，状态码：{response.status_code}"


def _repo_record(item, rank, detail: bool = False):
    """把 GitHub 仓库原始 dict 映射成统一的返回记录（不含涨星速度等扩展字段）。

    detail=True 额外带出 topics / forks / open_issues / license / created_at /
    pushed_at / homepage。对话榜单不需要这些（多字段会让模型啰嗦），
    但写文章时必须给：具体事实越多，句子越"有内容"，越不需要靠套话凑字数。
    """
    record = {
        "name": item["name"],
        "path": item["full_name"],
        "description": item["description"],
        "stars": item["stargazers_count"],
        "author": item["owner"]["login"],
        "rank": rank,
        "details": item["html_url"],
    }
    if detail:
        record.update({
            "topics": [t for t in (item.get("topics") or []) if t],
            "forks": item.get("forks_count"),
            "open_issues": item.get("open_issues_count"),
            "license": (item.get("license") or {}).get("spdx_id"),
            "created_at": item.get("created_at"),
            "pushed_at": item.get("pushed_at"),
            "homepage": item.get("homepage") or None,
        })
    return record


def _stars_per_day(item):
    """仓库平均每天新增星标数 = 星标总数 ÷ 已存在天数。

    created_at 形如 "2026-09-24T16:40:20Z"，用 UTC 当前时间算天数；
    不足 1 天按 1 天算，避免刚建的仓库速率虚高/除零。
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    created = datetime.datetime.fromisoformat(item["created_at"].replace("Z", "+00:00"))
    age_days = max(1, (now - created).days)
    return int(item["stargazers_count"]) / age_days


# ===== 公开取数函数：@tool 与"公众号文章"工具共用同一套排名逻辑 =====
# 抽出来的原因：文章工具如果让模型把刚查到的榜单"手抄"一遍，会漏字段、改星标数；
# 让它直接调用这两个函数取数，数据全程只经过一次程序化 json.dumps。

def fetch_hot_records(days_age: int = 7, language: str | None = None,
                      min_stars: int = 50, rank: int = 7, detail: bool = False):
    """取"星标总量"榜记录，返回 (records, error)；成功时 error 为 None。

    detail=True 时带上 topics/license/创建日期等扩展字段（文章工具用）。
    """
    days_age, min_stars, query = _build_recent_query(days_age, min_stars, language)

    # "排行榜"：按星标降序取前 rank 条；GitHub 搜索 API 单页上限 30，钳制在 [1, 30]。
    rank = max(1, min(int(rank), 30))
    items, error = _search_new_repos(query, per_page=rank)
    if error:
        return None, error

    return [_repo_record(item, i, detail) for i, item in enumerate(items, start=1)], None


def fetch_rising_records(days_age: int = 7, language: str | None = None,
                         min_stars: int = 10, rank: int = 7, detail: bool = False):
    """取"平均每天涨星速度"榜记录，返回 (records, error)；成功时 error 为 None。

    detail=True 时，_repo_record 已经带出 created_at，就不再重复追加。
    """
    days_age, min_stars, query = _build_recent_query(days_age, min_stars, language)
    rank = max(1, min(int(rank), 30))

    # API 没有"新增星标"限定符，先按星标降序取满 30 个候选（单页上限），
    # 真正的"涨星速度"排名在本地计算。
    items, error = _search_new_repos(query, per_page=30)
    if error:
        return None, error

    # 按"平均每天新增星标数"降序排名——这就是"最近 N 天新加最多 stars"的口径
    items.sort(key=_stars_per_day, reverse=True)

    records = []
    for i, item in enumerate(items[:rank], start=1):
        record = _repo_record(item, i, detail)
        record["stars_per_day"] = round(_stars_per_day(item), 1)
        if not detail:
            record["created_at"] = item["created_at"]
        records.append(record)
    return records, None


@tool
def get_github_rank(days_age: int = 7, language: str | None = None,
               min_stars: int = 50, rank: int = 7):
    """获取 GitHub 热门仓库信息。

    Args:
        days_age: 过去多少天内的热门仓库，整数，默认7天。
        language: 仓库语言过滤，例如 "Python"。默认不限制。
        min_stars: 仓库最少星标数，整数，默认100。
        rank: 返回排名前 rank 个仓库（按星标数降序），整数，默认7，上限30。
    Returns:
        name: GitHub 名字。
        path: 仓库路径。
        description: 仓库描述。
        stars: 仓库星标数。
        author: 仓库作者。
        rank: 仓库前rank条。
        details: 仓库详细信息。
    """

    try:
        records, error = fetch_hot_records(days_age, language, min_stars, rank)
        if error:
            return error
        return records

    except Exception as e:
        return (
            f"查找最近{days_age}天, 语言为{language or '所有语言'},最小星标数为{min_stars}的热门仓库时出错：{str(e)}"
        )


@tool
def get_github_recent_stars_rank(days_age: int = 7, language: str | None = None,
                                   min_stars: int = 10, rank: int = 7):
    """获取最近 N 天内新创建仓库的【新增星标数】排行榜（涨星最多排名）。

    与 get_github_rank 的区别：
    - get_github_rank 是"绝对星标数"排行：在时间窗口内挑星标总量最高的仓库；
    - 本工具是"平均涨星速度"排行：把 新增星标 ÷ 仓库已存在天数 作为涨星速率，
      按速率降序排名，能找出"这几天涨星最猛"的仓库（创建越晚、涨星越快排名越高）。

    Args:
        days_age: 统计最近多少天内新创建的仓库，整数，默认7天。
        language: 仓库语言过滤，例如 "Python"。默认不限制。
        min_stars: 仓库最少星标数，整数，默认10。
        rank: 返回排名前 rank 个仓库（按平均每天新增星标数降序），整数，默认7，上限30。
    Returns:
        name: GitHub 名字。
        path: 仓库路径。
        description: 仓库描述。
        stars: 仓库星标数（即最近 days_age 天内新增的星标数）。
        stars_per_day: 平均每天新增星标数（新增星标 ÷ 仓库已存在天数）。
        created_at: 仓库创建时间。
        author: 仓库作者。
        rank: 仓库前rank条。
        details: 仓库详细信息。
    """

    try:
        records, error = fetch_rising_records(days_age, language, min_stars, rank)
        if error:
            return error
        return records

    except Exception as e:
        return (
            f"查找最近{days_age}天新创建、语言为{language or '所有语言'}、最小星标数为{min_stars}的新星仓库排行榜时出错：{str(e)}"
        )
