# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Allowlisted, side-effect-free tools the Realtime demo proxy runs for the browser.

Any duplex model that publishes ``response.function_call_arguments.done`` can
use them; the proxy sends the output back as ``function_call_output``.
``WebSearch`` needs ``SERPER_API_KEY``; ``DeepSeek`` returns a mock without
``DEEPSEEK_API_KEY``. ``BraveSearch``, ``duckduckgo_search`` and ``WebFetch``
stay off unless ``VLLM_REALTIME_WEB_TOOL_BRAVE`` /
``VLLM_REALTIME_WEB_TOOL_DDG`` / ``VLLM_REALTIME_WEB_TOOL_WEBFETCH`` is set.

``VisionToolExecutor`` is a second tool set for camera-aware models. Two of its
search tools receive the latest camera frame, and every output is plain text.
"""

from __future__ import annotations

import ast
import asyncio
import datetime as dt
import ipaddress
import json
import logging
import math
import operator
import os
import socket
import time
from collections.abc import Callable, Iterator
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlparse

import regex as re
import requests
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

logger = logging.getLogger(__name__)

TOOL_TIMEOUT_SECONDS = 5.0
TOOL_OUTPUT_LIMIT_BYTES = 64 * 1024
SAFE_HTTP_TIMEOUT_SECONDS = 3.0
SAFE_HTTP_RESPONSE_LIMIT_BYTES = 256 * 1024
MAX_CALCULATOR_EXPRESSION_LENGTH = 200
MAX_CALCULATOR_AST_NODES = 64
MAX_CALCULATOR_ABS_VALUE = 1e100
MAX_CALCULATOR_EXPONENT = 100
DEEPSEEK_MAX_OUTPUT_CHARS = 8_000
DEEPSEEK_TIMEOUT_SECONDS = 30.0

_DEEPSEEK_SYSTEM_PROMPT = (
    "你是一个口语化、简洁的AI助手，必须优先使用联网搜索。"
    "默认每次回答前都先联网检索，能搜就搜，尽量不要跳过搜索。"
    "如果搜索失败，再基于已有知识给出答案，并简短说明未检索成功。"
    "用中文口语化回答，像朋友聊天一样自然。"
    "回复简短直接。不要列点、不要markdown格式。"
)
_CURRENCY_CODE_PATTERN = re.compile(r"^[A-Z]{3}$")
_CURRENCY_NAMES_ZH = {
    "CNY": "人民幣",
    "EUR": "歐元",
    "GBP": "英鎊",
    "HKD": "港幣",
    "JPY": "日圓",
    "USD": "美元",
}
_CALCULATOR_BINARY_OPERATORS: dict[type[ast.AST], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_CALCULATOR_UNARY_OPERATORS: dict[type[ast.AST], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_CALCULATOR_FUNCTIONS: dict[str, Callable[..., float]] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
}
_CALCULATOR_CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}
_WEATHER_LABELS_ZH = {
    0: "晴朗",
    1: "大致晴朗",
    2: "局部多雲",
    3: "陰天",
    45: "霧",
    48: "凍霧",
    51: "小毛毛雨",
    53: "中毛毛雨",
    55: "強毛毛雨",
    56: "小凍毛毛雨",
    57: "強凍毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    66: "小凍雨",
    67: "大凍雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    77: "米雪",
    80: "小陣雨",
    81: "中陣雨",
    82: "強陣雨",
    85: "小陣雪",
    86: "大陣雪",
    95: "雷暴",
    96: "雷暴伴小冰雹",
    99: "雷暴伴大冰雹",
}


class CalculatorArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    expression: str = Field(min_length=1, max_length=MAX_CALCULATOR_EXPRESSION_LENGTH)


class CurrentDatetimeArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class CityWeatherArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    city: str = Field(min_length=1, max_length=100)
    country: str = Field(default="", max_length=100)
    lang: str = Field(default="zh", pattern=r"^(zh|en)$")

    @field_validator("city", "country", "lang", mode="before")
    @classmethod
    def _strip_weather_strings(cls, value: Any) -> Any:
        # Models often pad values (``lang=" zh "``); strip instead of failing the call.
        if isinstance(value, str):
            return value.strip()
        return value


class CurrencyArguments(BaseModel):
    # Not strict: tag-style tool-call bodies often carry numbers as strings ("1").
    model_config = ConfigDict(extra="forbid")

    amount: float = Field(ge=0, le=1e12, allow_inf_nan=False)
    from_currency: str = Field(pattern=r"^[A-Za-z]{3}$")
    to_currency: str = Field(pattern=r"^[A-Za-z]{3}$")
    date: str = Field(default="", max_length=10)

    @field_validator("amount", mode="before")
    @classmethod
    def _coerce_amount(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("amount must be a number")
        if isinstance(value, str):
            value = value.strip().replace(",", "")
        return value

    @field_validator("from_currency", "to_currency", mode="before")
    @classmethod
    def _strip_currency_code(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("date", mode="before")
    @classmethod
    def _null_date_to_empty(cls, value: Any) -> Any:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        return value


class DeepSeekArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: str = Field(min_length=1, max_length=4096)
    enable_search: bool = True


class CurrentLocationArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class WebSearchArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: str = Field(min_length=1, max_length=256)
    max_results: int = Field(default=10, ge=1, le=20)


class DuckDuckGoArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: str = Field(min_length=1, max_length=256)
    max_results: int = Field(default=5, ge=1, le=10)


class WebFetchArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    url: str = Field(min_length=1, max_length=2048)


def _bounded_number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("calculator_non_numeric_result")
    result = float(value)
    if not math.isfinite(result) or abs(result) > MAX_CALCULATOR_ABS_VALUE:
        raise ValueError("calculator_result_out_of_range")
    return result


def _evaluate_calculator_node(node: ast.AST, *, depth: int = 0) -> float:
    if depth > 12:
        raise ValueError("calculator_expression_too_deep")
    if isinstance(node, ast.Constant):
        return _bounded_number(node.value)
    if isinstance(node, ast.Name):
        if node.id not in _CALCULATOR_CONSTANTS:
            raise ValueError("calculator_unknown_name")
        return _CALCULATOR_CONSTANTS[node.id]
    if isinstance(node, ast.UnaryOp):
        unary = _CALCULATOR_UNARY_OPERATORS.get(type(node.op))
        if unary is None:
            raise ValueError("calculator_unsupported_operator")
        return _bounded_number(unary(_evaluate_calculator_node(node.operand, depth=depth + 1)))
    if isinstance(node, ast.BinOp):
        binary = _CALCULATOR_BINARY_OPERATORS.get(type(node.op))
        if binary is None:
            raise ValueError("calculator_unsupported_operator")
        left = _evaluate_calculator_node(node.left, depth=depth + 1)
        right = _evaluate_calculator_node(node.right, depth=depth + 1)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_CALCULATOR_EXPONENT:
            raise ValueError("calculator_exponent_out_of_range")
        return _bounded_number(binary(left, right))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _CALCULATOR_FUNCTIONS:
            raise ValueError("calculator_unsupported_function")
        if node.keywords or not 1 <= len(node.args) <= 8:
            raise ValueError("calculator_invalid_function_arguments")
        arguments: list[Any] = [_evaluate_calculator_node(arg, depth=depth + 1) for arg in node.args]
        if node.func.id == "round" and len(arguments) == 2:
            if not arguments[1].is_integer() or not -15 <= arguments[1] <= 15:
                raise ValueError("calculator_invalid_round_digits")
            arguments[1] = int(arguments[1])
        return _bounded_number(_CALCULATOR_FUNCTIONS[node.func.id](*arguments))
    raise ValueError("calculator_unsupported_expression")


def _calculator(arguments: CalculatorArguments) -> dict[str, Any]:
    expression = arguments.expression.strip()
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("calculator_invalid_syntax") from exc
    if sum(1 for _ in ast.walk(tree)) > MAX_CALCULATOR_AST_NODES:
        raise ValueError("calculator_expression_too_complex")
    result = _evaluate_calculator_node(tree.body)
    rendered: int | float = int(result) if result.is_integer() else result
    return {"expression": expression, "result": rendered, "summary": f"{expression} = {rendered}"}


def _current_datetime(arguments: CurrentDatetimeArguments) -> dict[str, Any]:
    del arguments
    timezone = dt.timezone(dt.timedelta(hours=8), name="Asia/Shanghai")
    now = dt.datetime.now(timezone)
    return {
        "datetime": now.isoformat(timespec="seconds"),
        "timezone": "Asia/Shanghai",
        "summary": (
            f"目前上海時間是{now.year}年{now.month}月{now.day}日{now.hour}時{now.minute}分{now.second}秒（UTC+8）。"
        ),
    }


def _bounded_json(response: requests.Response) -> dict[str, Any]:
    response.raise_for_status()
    if len(response.content) > SAFE_HTTP_RESPONSE_LIMIT_BYTES:
        raise ValueError("upstream_response_too_large")
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("upstream_invalid_response")
    return payload


def _get_bounded_json(url: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
    return _bounded_json(requests.get(url, params=params, timeout=SAFE_HTTP_TIMEOUT_SECONDS))


def _post_bounded_json(
    url: str,
    *,
    json_body: dict[str, Any],
    headers: dict[str, str] | None = None,
    timeout_seconds: float = SAFE_HTTP_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    return _bounded_json(requests.post(url, json=json_body, headers=headers, timeout=timeout_seconds))


def _city_weather(arguments: CityWeatherArguments) -> dict[str, Any]:
    city = arguments.city.strip()
    country = arguments.country.strip().casefold()
    geocoding = _get_bounded_json(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 10, "language": arguments.lang, "format": "json"},
    )
    candidates = geocoding.get("results")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("weather_city_not_found")
    location = candidates[0]
    if country:
        location = next(
            (
                item
                for item in candidates
                if isinstance(item, dict)
                and country in {str(item.get("country", "")).casefold(), str(item.get("country_code", "")).casefold()}
            ),
            location,
        )
    if not isinstance(location, dict):
        raise ValueError("weather_invalid_location")
    latitude = location.get("latitude")
    longitude = location.get("longitude")
    if not isinstance(latitude, (int, float)) or not isinstance(longitude, (int, float)):
        raise ValueError("weather_missing_coordinates")
    weather = _get_bounded_json(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": (
                "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m"
            ),
            "timezone": "auto",
        },
    )
    current = weather.get("current")
    units = weather.get("current_units")
    if not isinstance(current, dict):
        raise ValueError("weather_current_unavailable")
    if not isinstance(units, dict):
        units = {}
    code = int(current.get("weather_code", -1))
    result = {
        "city": location.get("name", city),
        "country": location.get("country", ""),
        "region": location.get("admin1", ""),
        "timezone": weather.get("timezone", ""),
        "time": current.get("time", ""),
        "condition": _WEATHER_LABELS_ZH.get(code, f"未知天氣代碼 {code}"),
        "temperature": current.get("temperature_2m"),
        "temperature_unit": units.get("temperature_2m", ""),
        "feels_like": current.get("apparent_temperature"),
        "humidity": current.get("relative_humidity_2m"),
        "humidity_unit": units.get("relative_humidity_2m", ""),
        "precipitation": current.get("precipitation"),
        "precipitation_unit": units.get("precipitation", ""),
        "wind_speed": current.get("wind_speed_10m"),
        "wind_speed_unit": units.get("wind_speed_10m", ""),
        "source": "open-meteo.com",
    }
    result["summary"] = (
        f"{result['city']}目前{result['condition']}，氣溫{result['temperature']}"
        f"{result['temperature_unit']}，體感{result['feels_like']}{result['temperature_unit']}。"
    )
    return result


def _convert_currency(arguments: CurrencyArguments) -> dict[str, Any]:
    source = arguments.from_currency.upper()
    target = arguments.to_currency.upper()
    if not _CURRENCY_CODE_PATTERN.fullmatch(source) or not _CURRENCY_CODE_PATTERN.fullmatch(target):
        raise ValueError("currency_invalid_code")
    date = arguments.date.strip()
    if date:
        try:
            dt.date.fromisoformat(date)
        except ValueError as exc:
            raise ValueError("currency_invalid_date") from exc
    source_zh = _CURRENCY_NAMES_ZH.get(source, source)
    target_zh = _CURRENCY_NAMES_ZH.get(target, target)
    if source == target:
        return {
            "amount": arguments.amount,
            "from_currency": source,
            "to_currency": target,
            "converted_amount": arguments.amount,
            "rate": 1.0,
            "date": date or "latest",
            "source": "identity",
            "summary": (
                f"{arguments.amount:g} {source_zh}等於{arguments.amount:g} {target_zh}"
                f"（{arguments.amount:g} {source} = {arguments.amount:g} {target}）。"
            ),
        }
    payload = _get_bounded_json(
        f"https://api.frankfurter.app/{date or 'latest'}",
        params={"amount": arguments.amount, "from": source, "to": target},
    )
    rates = payload.get("rates")
    if not isinstance(rates, dict) or not isinstance(rates.get(target), (int, float)):
        raise ValueError("currency_rate_unavailable")
    converted = _bounded_number(rates[target])
    return {
        "amount": arguments.amount,
        "from_currency": source,
        "to_currency": target,
        "converted_amount": converted,
        "rate": round(converted / arguments.amount, 10) if arguments.amount else 0.0,
        "date": payload.get("date", date or "latest"),
        "source": "frankfurter.app",
        "summary": (
            f"{arguments.amount:g} {source_zh}約等於{converted:g} {target_zh}"
            f"（{arguments.amount:g} {source} = {converted:g} {target}）。"
        ),
    }


def _deepseek(arguments: DeepSeekArguments) -> dict[str, Any]:
    query = arguments.query.strip()
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        summary = f"DeepSeek mock：已收到問題「{query}」，未呼叫真實搜尋。"
        return {"query": query, "enable_search": arguments.enable_search, "mocked": True, "summary": summary}
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/anthropic").rstrip("/")
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat").strip() or "deepseek-chat"
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": 256,
        "temperature": 0.7,
        "system": _DEEPSEEK_SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": query}],
    }
    if arguments.enable_search:
        body["tools"] = [{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}]
    try:
        payload = _post_bounded_json(
            f"{base_url}/v1/messages",
            json_body=body,
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            timeout_seconds=DEEPSEEK_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise ValueError("deepseek_upstream_failed") from exc
    content = payload.get("content")
    if not isinstance(content, list):
        raise ValueError("deepseek_invalid_response")
    parts = [
        str(block.get("text") or "").strip()
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    text = "\n".join(part for part in parts if part).strip()
    if not text:
        raise ValueError("deepseek_empty_response")
    text = text[:DEEPSEEK_MAX_OUTPUT_CHARS]
    return {"query": query, "enable_search": arguments.enable_search, "mocked": False, "model": model, "summary": text}


def _location_from_ipwho(payload: dict[str, Any]) -> dict[str, Any]:
    if not payload.get("success", False):
        raise ValueError("location_ipwho_failed")
    timezone = payload.get("timezone")
    connection = payload.get("connection")
    return {
        "public_ip": payload.get("ip", ""),
        "country": payload.get("country", ""),
        "region": payload.get("region", ""),
        "city": payload.get("city", ""),
        "latitude": payload.get("latitude"),
        "longitude": payload.get("longitude"),
        "timezone": timezone.get("id", "") if isinstance(timezone, dict) else "",
        "isp": connection.get("isp", "") if isinstance(connection, dict) else "",
        "source": "ipwho.is",
        "accuracy_note": "Approximate location inferred from public IP.",
    }


def _location_from_ipapi(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("error"):
        raise ValueError("location_ipapi_failed")
    return {
        "public_ip": payload.get("ip", ""),
        "country": payload.get("country_name", ""),
        "region": payload.get("region", ""),
        "city": payload.get("city", ""),
        "latitude": payload.get("latitude"),
        "longitude": payload.get("longitude"),
        "timezone": payload.get("timezone", ""),
        "isp": payload.get("org", ""),
        "source": "ipapi.co",
        "accuracy_note": "Approximate location inferred from public IP.",
    }


def _get_current_location(arguments: CurrentLocationArguments) -> dict[str, Any]:
    del arguments
    services: tuple[tuple[str, Callable[[dict[str, Any]], dict[str, Any]]], ...] = (
        ("https://ipwho.is/", _location_from_ipwho),
        ("https://ipapi.co/json/", _location_from_ipapi),
    )
    for url, parse in services:
        try:
            result = parse(_get_bounded_json(url))
        except (ValueError, requests.RequestException):
            continue
        place = "、".join(
            str(part) for part in (result.get("city"), result.get("region"), result.get("country")) if part
        )
        result["summary"] = f"目前大約在{place}。" if place else "已取得目前大約位置（依公開 IP 推估）。"
        return result
    raise ValueError("location_lookup_failed")


def _search_result(query: str, max_results: int, source: str, label: str, results: list[dict[str, str]]):
    summary = f"{label}「{query}」找到{len(results)}筆結果。" if results else f"{label}「{query}」沒有結果。"
    if results and results[0].get("title"):
        summary = f"{summary}首筆：{results[0]['title']}"
    return {"query": query, "max_results": max_results, "source": source, "results": results, "summary": summary}


def _web_search(arguments: WebSearchArguments) -> dict[str, Any]:
    api_key = os.environ.get("SERPER_API_KEY", "").strip()
    if not api_key:
        raise ValueError("search_api_key_missing")
    query = arguments.query.strip()
    endpoint = os.environ.get("SERPER_SEARCH_ENDPOINT", "").strip() or "https://google.serper.dev/search"
    try:
        payload = _post_bounded_json(
            endpoint,
            json_body={"q": query, "num": arguments.max_results},
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        )
    except requests.RequestException as exc:
        raise ValueError("search_upstream_failed") from exc
    raw = payload.get("organic") or []
    if not isinstance(raw, list):
        raise ValueError("search_invalid_response")
    results = [
        {
            "title": str(item.get("title", ""))[:500],
            "snippet": str(item.get("snippet", ""))[:500],
            "link": str(item.get("link", ""))[:500],
        }
        for item in raw[: arguments.max_results]
        if isinstance(item, dict)
    ]
    return _search_result(query, arguments.max_results, "serper", "搜尋", results)


def _brave_search(arguments: WebSearchArguments) -> dict[str, Any]:
    api_key = os.environ.get("BRAVE_SEARCH_API_KEY", "").strip()
    if not api_key:
        raise ValueError("search_api_key_missing")
    query = arguments.query.strip()
    endpoint = os.environ.get("BRAVE_SEARCH_ENDPOINT", "").strip() or "https://api.search.brave.com/res/v1/web/search"
    try:
        payload = _bounded_json(
            requests.get(
                endpoint,
                headers={"Accept": "application/json", "X-Subscription-Token": api_key},
                params={"q": query, "count": arguments.max_results},
                timeout=SAFE_HTTP_TIMEOUT_SECONDS,
            )
        )
    except requests.RequestException as exc:
        raise ValueError("search_upstream_failed") from exc
    raw = (payload.get("web") or {}).get("results") or []
    if not isinstance(raw, list):
        raise ValueError("search_invalid_response")
    results = [
        {
            "title": str(item.get("title", ""))[:500],
            "snippet": str(item.get("description", ""))[:500],
            "link": str(item.get("url", ""))[:500],
        }
        for item in raw[: arguments.max_results]
        if isinstance(item, dict)
    ]
    return _search_result(query, arguments.max_results, "brave", "Brave 搜尋", results)


class _DuckDuckGoHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._in_link = False
        self._in_snippet = False
        self._skip_snippet = False
        self._current: dict[str, str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        cls = attrs_dict.get("class") or ""
        if tag == "a" and "result__a" in cls:
            self._in_link = True
            self._current = {"title": "", "snippet": "", "url": str(attrs_dict.get("href") or "")}
            return
        if self._current is not None and tag == "a" and "badge" in cls:
            self._skip_snippet = True
            return
        if tag == "td" and "result__snippet" in cls:
            self._in_snippet = True

    def handle_data(self, data: str) -> None:
        if self._current is None:
            return
        if self._in_link:
            self._current["title"] += data
        elif self._in_snippet and not self._skip_snippet:
            self._current["snippet"] += data

    def handle_endtag(self, tag: str) -> None:
        if self._in_link and tag == "a":
            self._in_link = False
        if self._in_snippet and tag == "td":
            self._in_snippet = False
        if not self._in_link and not self._in_snippet and self._current is not None:
            if self._current["title"] or self._current["snippet"]:
                self.results.append(dict(self._current))
            self._current = None
            self._skip_snippet = False


def _duckduckgo_search(arguments: DuckDuckGoArguments) -> dict[str, Any]:
    query = arguments.query.strip()
    try:
        response = requests.get(
            "https://html.duckduckgo.com/html/", params={"q": query}, timeout=SAFE_HTTP_TIMEOUT_SECONDS
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ValueError("search_upstream_failed") from exc
    if len(response.content) > SAFE_HTTP_RESPONSE_LIMIT_BYTES:
        raise ValueError("upstream_response_too_large")
    parser = _DuckDuckGoHTMLParser()
    parser.feed(response.text)
    results = [
        {
            "title": str(item.get("title", ""))[:500],
            "snippet": str(item.get("snippet", ""))[:500],
            "link": str(item.get("url", ""))[:500],
        }
        for item in parser.results[: arguments.max_results]
    ]
    return _search_result(query, arguments.max_results, "duckduckgo", "DuckDuckGo 搜尋", results)


def _check_ssrf(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("webfetch_invalid_url")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise ValueError("webfetch_dns_failed") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_unspecified
            or ip.is_reserved
            or ip.is_multicast
        ):
            raise ValueError("webfetch_ssrf_blocked")
    return url


def _web_fetch(arguments: WebFetchArguments) -> dict[str, Any]:
    url = _check_ssrf(arguments.url.strip().strip("\"'`“”‘’"))
    try:
        response = requests.get(
            url,
            timeout=SAFE_HTTP_TIMEOUT_SECONDS,
            allow_redirects=False,
            headers={"User-Agent": "vLLM-Omni-Realtime-WebFetch/1.0"},
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ValueError("webfetch_upstream_failed") from exc
    if len(response.content) > SAFE_HTTP_RESPONSE_LIMIT_BYTES:
        raise ValueError("upstream_response_too_large")
    content_type = str(response.headers.get("content-type") or "")
    if any(kind in content_type for kind in ("html", "xml", "text", "json")):
        text = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", response.text)
        text = re.sub(r"(?s)<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()[:8000]
    else:
        text = f"Binary or non-text response ({content_type}, {len(response.content)} bytes)."
    return {"url": url, "content_type": content_type, "text": text, "summary": text[:240] if text else f"已取得 {url}"}


_SAFE_TOOL_DEFINITIONS: tuple[tuple[str, str, type[BaseModel], Callable[[Any], dict[str, Any]]], ...] = (
    (
        "calculator",
        "Safely evaluate arithmetic. Treat its result as final; do not call another tool to verify it.",
        CalculatorArguments,
        _calculator,
    ),
    (
        "get_current_datetime",
        "Return the current Asia/Shanghai date and time. Treat its result as final and answer without other tools.",
        CurrentDatetimeArguments,
        _current_datetime,
    ),
    (
        "get_city_weather",
        "Look up current city weather using Open-Meteo. Treat its result as final; do not call another tool.",
        CityWeatherArguments,
        _city_weather,
    ),
    (
        "convert_currency",
        "Convert an amount between ISO currencies. The converted amount is final; do not recalculate it.",
        CurrencyArguments,
        _convert_currency,
    ),
    (
        "DeepSeek",
        "General assistant for open questions. Uses live DeepSeek with DEEPSEEK_API_KEY; otherwise returns a mock.",
        DeepSeekArguments,
        _deepseek,
    ),
    (
        "get_current_location",
        "Get the approximate geolocation from the public IP. Treat the result as final.",
        CurrentLocationArguments,
        _get_current_location,
    ),
    (
        "WebSearch",
        "General-purpose web search. Use for factual lookup and recent events. Treat ranked results as final.",
        WebSearchArguments,
        _web_search,
    ),
)

_OPTIONAL_SAFE_TOOL_DEFINITIONS: tuple[tuple[str, str, str, type[BaseModel], Callable[[Any], dict[str, Any]]], ...] = (
    (
        "VLLM_REALTIME_WEB_TOOL_BRAVE",
        "BraveSearch",
        "Alternative web search via Brave Search API. Prefer as a second source when Serper is unavailable.",
        WebSearchArguments,
        _brave_search,
    ),
    (
        "VLLM_REALTIME_WEB_TOOL_DDG",
        "duckduckgo_search",
        "Search the web via DuckDuckGo HTML results. Fallback / alternative search engine.",
        DuckDuckGoArguments,
        _duckduckgo_search,
    ),
    (
        "VLLM_REALTIME_WEB_TOOL_WEBFETCH",
        "WebFetch",
        "Fetch a public http(s) URL and return cleaned text. Private/loopback hosts are blocked.",
        WebFetchArguments,
        _web_fetch,
    ),
)

_WEATHER_ALIASES = {
    "get_weather",
    "weather",
    "city_weather",
    "getcityweather",
    "cityweather",
    "lookup_weather",
    "查询天气",
    "查天气",
    "天气查询",
}


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _iter_tool_definitions() -> Iterator[tuple[str, str, type[BaseModel], Callable[[Any], dict[str, Any]]]]:
    yield from _SAFE_TOOL_DEFINITIONS
    for env_name, name, description, arguments_model, handler in _OPTIONAL_SAFE_TOOL_DEFINITIONS:
        if _env_enabled(env_name):
            yield name, description, arguments_model, handler


def _canonical_tool_name(name: str) -> str:
    """Map common model aliases for the weather tool onto its registered name."""
    raw = (name or "").strip()
    lowered = raw.lower().replace(" ", "_").replace("-", "_")
    collapsed = "".join(ch for ch in lowered if ch.isalnum())
    if raw in _WEATHER_ALIASES or lowered in _WEATHER_ALIASES or collapsed in _WEATHER_ALIASES:
        return "get_city_weather"
    if "weather" in collapsed and "location" not in collapsed:
        return "get_city_weather"
    return raw


def _error_output(code: str) -> str:
    return json.dumps({"ok": False, "error": {"code": code}}, separators=(",", ":"))


class RealtimeToolExecutor:
    """Run the allowlisted tools; one ``run`` per published function call."""

    def __init__(self) -> None:
        self._registry = {
            name: (description, arguments_model, handler)
            for name, description, arguments_model, handler in _iter_tool_definitions()
        }

    @property
    def realtime_tools(self) -> list[dict[str, Any]]:
        """Tool objects in the flat Realtime ``session.tools`` shape."""
        return [
            {
                "type": "function",
                "name": name,
                "description": description,
                "parameters": arguments_model.model_json_schema(),
            }
            for name, (description, arguments_model, _handler) in self._registry.items()
        ]

    async def run(self, name: str, arguments: str) -> str:
        """Return the ``function_call_output`` string; failures become ``{"ok": false}``."""
        started = time.perf_counter()
        tool_name = _canonical_tool_name(name)
        status = "error"
        try:
            entry = self._registry.get(tool_name)
            if entry is None:
                raise ValueError("unknown_tool")
            _description, arguments_model, handler = entry
            try:
                parsed = json.loads(arguments or "{}")
                validated = arguments_model.model_validate(parsed)
            except (json.JSONDecodeError, ValidationError) as exc:
                raise ValueError("invalid_tool_arguments") from exc
            timeout_seconds = (
                max(TOOL_TIMEOUT_SECONDS, DEEPSEEK_TIMEOUT_SECONDS)
                if tool_name == "DeepSeek" and os.environ.get("DEEPSEEK_API_KEY", "").strip()
                else TOOL_TIMEOUT_SECONDS
            )
            value = await asyncio.wait_for(asyncio.to_thread(handler, validated), timeout=timeout_seconds)
            output = json.dumps({"ok": True, "result": value}, ensure_ascii=False, separators=(",", ":"))
            if len(output.encode("utf-8")) > TOOL_OUTPUT_LIMIT_BYTES:
                raise ValueError("tool_output_too_large")
            status = "completed"
        except (TimeoutError, asyncio.TimeoutError):
            output = _error_output("tool_timeout")
        except ValueError as exc:
            output = _error_output(str(exc))
        except Exception:
            logger.exception("Realtime tool %s failed", tool_name)
            output = _error_output("tool_execution_failed")
        logger.info(
            "Realtime tool %s status=%s latency_ms=%.1f output_bytes=%d",
            tool_name,
            status,
            (time.perf_counter() - started) * 1000,
            len(output.encode("utf-8")),
        )
        return output


# ---- Vision tool set ----
#
# Schemas are kept verbatim and in this order. Looser descriptions or extra
# search tools make a camera-aware model call search for requests it should
# answer from the frame, such as drawing a box. Each tool returns plain text,
# which goes back to the model unwrapped; errors are plain text too
# (``tool not found: <name>``, ``tool timeout``, the exception message).

_VISION_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "deepseek_vlm_search",
            "description": "结合当前画面，检索画面中清晰可见对象需要联网或外部知识才能确认的信息，例如公开名称、身份和关键背景资料。系统会把发起调用那一刻的当前画面自动当作输入，无需也不要传图片参数。前提：必须等画面中的目标已经清晰可见、对焦稳定后再调用；目标模糊、晃动、被遮挡或尚未对准时禁止调用，先等待画面清晰。仅在回答必须依赖这类额外网络知识时才可调用；可凭自身视觉直接回答时禁止调用。返回结果仅供参考，以你自己的视觉判断为准。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "需要对照当前清晰画面才能对应到的外部或实时信息问题。目标尚未清晰可见时不要调用；禁止写成「帮我看清」之类的纯视觉问题。",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "description": "从网络时间服务读取当前北京日期和时间，返回带时区的 ISO 8601。仅在回答必须依赖当前真实日期或时刻这一实时信息时才可调用，例如用户明确问现在几点、今天几号。禁止因为画面看不清、闲聊或可凭常识回答而调用。返回结果仅供参考，以你自己的判断为准。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "DeepSeek",
            "description": "按文本在网上检索外部知识或实时信息。不看画面、也不识图，只接受文字问题。只在回答必须依赖联网文字资料时使用；自己已经知道的不必查。结果仅供参考。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "要检索的文本问题。只接受文字，不要写入需要看图才能问的内容。",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_location",
            "description": "根据公网 IP 查询大致地理位置（城市、地区、国家等），仅城市级精度，无法精确到街道、建筑或画面中的地点。仅在回答必须依赖用户当前所在城市或地区这一额外信息时才可调用。禁止用本工具判断画面里是什么地方——地点外观必须依赖你自己的视觉能力。不要因为看不清或想确认自己的判断而调用。返回结果仅供参考，以你自己的判断为准。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_city_weather",
            "description": "查询指定城市的实时或指定日期天气，返回温度、天气状况、降水、风速等实时气象数据。仅在用户明确询问天气、气温、降水等且必须依赖这类实时信息时才可调用。禁止主动或顺便查询；画面中的天色、是否在下雨必须先用你自己的视觉判断。用户未给出城市时，需要城市名称才能查询。返回结果仅供参考，以你自己的判断为准。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name, e.g. Shanghai, Singapore, Paris."},
                    "country": {"type": "string", "description": "Optional country name/code to disambiguate."},
                    "lang": {"type": "string", "description": "Language hint, e.g. zh or en. Default zh."},
                    "date": {"type": "string", "description": "Optional date in YYYY-MM-DD format. 不传则查实时天气。"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "visual_web_search",
            "description": "辅助检索：把当前最新画面和文本问题一起提交到互联网搜索，返回一段文本答案。调用时自动附带那一刻的画面，不必也不要传入图片。等目标清晰可见、对焦稳定后再调用。画面模糊、晃动、被遮挡或尚未对准时不要调用，先等画面清楚。只在答案取决于画面中的对象，且该事实必须上网才能确认时调用，例如公开名称、身份、地点、商品或实时背景。凭自己看图或已有知识就能回答时不要调用。结果仅供参考，画面仍以你自己看到的为准。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "连同当前清晰画面一起上网检索的具体问题。写明要确认的外部事实，例如「画面中这家店的公开名称和主营」。目标尚未清晰入画时不要调用。不要写成只要求看清画面的问题。",
                    }
                },
                "required": ["query"],
            },
        },
    },
]

_BEIJING = dt.timezone(dt.timedelta(hours=8))
_DEEPSEEK_URL = "https://api.deepseek.com/anthropic/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"
_DEEPSEEK_CHAT_URL = "https://api.deepseek.com/chat/completions"

_WEATHER_CODES = {
    0: ("Clear sky", "晴朗"),
    1: ("Mainly clear", "大体晴朗"),
    2: ("Partly cloudy", "局部多云"),
    3: ("Overcast", "阴天"),
    45: ("Fog", "雾"),
    48: ("Depositing rime fog", "冻雾"),
    51: ("Light drizzle", "小毛毛雨"),
    53: ("Moderate drizzle", "中毛毛雨"),
    55: ("Dense drizzle", "强毛毛雨"),
    56: ("Light freezing drizzle", "小冻毛毛雨"),
    57: ("Dense freezing drizzle", "强冻毛毛雨"),
    61: ("Slight rain", "小雨"),
    63: ("Moderate rain", "中雨"),
    65: ("Heavy rain", "大雨"),
    66: ("Light freezing rain", "小冻雨"),
    67: ("Heavy freezing rain", "强冻雨"),
    71: ("Slight snow fall", "小雪"),
    73: ("Moderate snow fall", "中雪"),
    75: ("Heavy snow fall", "大雪"),
    77: ("Snow grains", "米雪"),
    80: ("Slight rain showers", "小阵雨"),
    81: ("Moderate rain showers", "中阵雨"),
    82: ("Violent rain showers", "强阵雨"),
    85: ("Slight snow showers", "小阵雪"),
    86: ("Heavy snow showers", "强阵雪"),
    95: ("Thunderstorm", "雷暴"),
    96: ("Thunderstorm with slight hail", "伴有小冰雹的雷暴"),
    99: ("Thunderstorm with heavy hail", "伴有强冰雹的雷暴"),
}

_DEEPSEEK_PROMPT = (
    "你是联网文字检索助手。针对用户的文本问题，检索并回答需要上网才能确认的事实或实时信息。"
    "用中文简短作答，不要列点、不要 markdown。"
    "查不到或无法确认就直说，不要编造。"
)
_VISUAL_PROMPT = (
    "你是通用图搜助手。你会同时看到当前画面和一条检索问题。"
    "先确认问题指向画面里哪一个清晰可见的对象，再联网检索这个对象需要外部资料才能确定的事实，并据此作答。"
    "用中文简洁准确回答，不要列点，不要 markdown。"
    "看不清、问题和画面对不上、或网上查不到时直接说明。不要编造，也不要只描写外观。"
)
_VLM_PROMPT = (
    "你是通用图搜助手。结合当前画面，检索画面中清晰可见对象需要联网或外部知识才能确认的信息，"
    "例如公开名称、身份和关键背景资料。"
    "用中文简洁准确作答。看不清或无法确认的细节不要编造，也不要只复述外观。"
)


def _dated_prompt(prompt: str) -> str:
    now = dt.datetime.now(_BEIJING)
    return f"当前日期时间：{now.strftime('%Y-%m-%d %H:%M')}（北京时间）\n\n{prompt}"


def _deepseek_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise ValueError("DeepSeek API key is not configured")
    return key


def _anthropic_text(data: dict[str, Any]) -> str:
    parts = [
        str(block.get("text") or "").strip()
        for block in data.get("content") or []
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "\n".join(part for part in parts if part).strip()


def _request_error_detail(exc: requests.RequestException) -> str:
    response = getattr(exc, "response", None)
    if response is None:
        return str(exc)
    body = (response.text or "").strip()
    if len(body) > 500:
        body = body[:500] + "..."
    return f"{exc}; body={body}" if body else str(exc)


def get_current_datetime() -> str:
    try:
        resp = requests.get("https://worldtimeapi.org/api/timezone/Asia/Shanghai", timeout=5)
        resp.raise_for_status()
        return resp.json()["datetime"]
    except Exception:
        return dt.datetime.now(_BEIJING).isoformat(timespec="seconds")


def ask_deepseek(query: str) -> str:
    q = (query or "").strip()
    if not q:
        return "Error: query is required"
    resp = requests.post(
        _DEEPSEEK_URL,
        headers={"x-api-key": _deepseek_key(), "anthropic-version": _ANTHROPIC_VERSION},
        json={
            "model": "deepseek-chat",
            "max_tokens": 256,
            "temperature": 0.7,
            "system": _dated_prompt(_DEEPSEEK_PROMPT),
            "messages": [{"role": "user", "content": q}],
            "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}],
        },
        timeout=60,
    )
    resp.raise_for_status()
    text = _anthropic_text(resp.json())
    if not text:
        return "DeepSeek returned no text content."
    if len(text) > 4096:
        text = text[:4096].rstrip() + "…"
    return text


def _from_ipwho(payload: dict) -> dict:
    if not payload.get("success", False):
        raise ValueError(payload.get("message", "ipwho.is lookup failed"))
    return {
        "public_ip": payload.get("ip", ""),
        "country": payload.get("country", ""),
        "region": payload.get("region", ""),
        "city": payload.get("city", ""),
        "latitude": payload.get("latitude"),
        "longitude": payload.get("longitude"),
        "timezone": payload.get("timezone", {}).get("id", ""),
        "isp": payload.get("connection", {}).get("isp", ""),
        "source": "ipwho.is",
        "accuracy_note": "Approximate location inferred from public IP.",
    }


def _from_ipapi(payload: dict) -> dict:
    if payload.get("error"):
        raise ValueError(payload.get("reason", "ipapi.co lookup failed"))
    return {
        "public_ip": payload.get("ip", ""),
        "country": payload.get("country_name", ""),
        "region": payload.get("region", ""),
        "city": payload.get("city", ""),
        "latitude": payload.get("latitude"),
        "longitude": payload.get("longitude"),
        "timezone": payload.get("timezone", ""),
        "isp": payload.get("org", ""),
        "source": "ipapi.co",
        "accuracy_note": "Approximate location inferred from public IP.",
    }


def get_current_location() -> str:
    errors: list[str] = []
    for url, parser in (("https://ipwho.is/", _from_ipwho), ("https://ipapi.co/json/", _from_ipapi)):
        try:
            resp = requests.get(url, timeout=6)
            resp.raise_for_status()
            return json.dumps(parser(resp.json()), ensure_ascii=False)
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    return "Location lookup failed: " + " | ".join(errors)


def _weather_label(code: int, lang: str) -> str:
    info = _WEATHER_CODES.get(code)
    if info is None:
        return "Unknown weather"
    return info[1] if lang.lower().startswith("zh") else info[0]


def get_city_weather(city: str, country: str = "", lang: str = "zh", date: str = "") -> str:
    city_name = (city or "").strip()
    country_name = (country or "").strip().lower()
    language = (lang or "zh").strip() or "zh"
    date_str = (date or "").strip()
    if not city_name:
        return "Error: city is required"
    try:
        geo_resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city_name, "count": 10, "language": language, "format": "json"},
            timeout=10,
        )
        geo_resp.raise_for_status()
        candidates = geo_resp.json().get("results") or []
        if not candidates:
            return f"Weather lookup failed: city not found ({city_name})"
        location = candidates[0]
        if country_name:
            for item in candidates:
                if country_name in {str(item.get("country", "")).lower(), str(item.get("country_code", "")).lower()}:
                    location = item
                    break
        lat = location.get("latitude")
        lon = location.get("longitude")
        if lat is None or lon is None:
            return "Weather lookup failed: missing coordinates"

        if date_str:
            try:
                query_date = dt.date.fromisoformat(date_str)
            except ValueError:
                return "日期格式错误，请使用 YYYY-MM-DD"
            api_base = (
                "https://api.open-meteo.com/v1/forecast"
                if query_date > dt.date.today()
                else "https://archive-api.open-meteo.com/v1/archive"
            )
            weather_resp = requests.get(
                api_base,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "daily": "temperature_2m_max,temperature_2m_min,weather_code,precipitation_sum,wind_speed_10m_max",
                    "start_date": date_str,
                    "end_date": date_str,
                    "timezone": "auto",
                },
                timeout=10,
            )
            weather_resp.raise_for_status()
            weather_data = weather_resp.json()
            daily = weather_data.get("daily") or {}
            units = weather_data.get("daily_units") or {}
            if not daily.get("time", []):
                return f"未找到 {date_str} 的天气数据"
            code = int((daily.get("weather_code") or [None])[0] or -1)
            return json.dumps(
                {
                    "city": location.get("name", city_name),
                    "country": location.get("country", ""),
                    "region": location.get("admin1", ""),
                    "latitude": lat,
                    "longitude": lon,
                    "timezone": weather_data.get("timezone", ""),
                    "date": date_str,
                    "weather_code": code,
                    "weather_text": _weather_label(code, language),
                    "temperature_max": (daily.get("temperature_2m_max") or [None])[0],
                    "temperature_min": (daily.get("temperature_2m_min") or [None])[0],
                    "temperature_unit": units.get("temperature_2m_max", ""),
                    "precipitation": (daily.get("precipitation_sum") or [None])[0],
                    "precipitation_unit": units.get("precipitation_sum", ""),
                    "wind_speed": (daily.get("wind_speed_10m_max") or [None])[0],
                    "wind_speed_unit": units.get("wind_speed_10m_max", ""),
                    "source": "open-meteo.com",
                },
                ensure_ascii=False,
            )

        weather_resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": (
                    "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m"
                ),
                "timezone": "auto",
            },
            timeout=10,
        )
        weather_resp.raise_for_status()
        weather_data = weather_resp.json()
        current = weather_data.get("current") or {}
        units = weather_data.get("current_units") or {}
        code = int(current.get("weather_code", -1))
        return json.dumps(
            {
                "city": location.get("name", city_name),
                "country": location.get("country", ""),
                "region": location.get("admin1", ""),
                "latitude": lat,
                "longitude": lon,
                "timezone": weather_data.get("timezone", ""),
                "time": current.get("time", ""),
                "weather_code": code,
                "weather_text": _weather_label(code, language),
                "temperature": current.get("temperature_2m"),
                "temperature_unit": units.get("temperature_2m", ""),
                "feels_like": current.get("apparent_temperature"),
                "humidity": current.get("relative_humidity_2m"),
                "humidity_unit": units.get("relative_humidity_2m", ""),
                "precipitation": current.get("precipitation"),
                "precipitation_unit": units.get("precipitation", ""),
                "wind_speed": current.get("wind_speed_10m"),
                "wind_speed_unit": units.get("wind_speed_10m", ""),
                "source": "open-meteo.com",
            },
            ensure_ascii=False,
        )
    except Exception as exc:
        return f"Weather lookup failed: {exc}"


def visual_web_search(query: str, image: str = "") -> str:
    q = (query or "").strip()
    if not q:
        return "visual_web_search failed: query is required."
    if not (image or "").strip():
        return (
            "visual_web_search failed: no video frame or image has been received yet, so there is no image to search."
        )
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        return "visual_web_search failed: set DEEPSEEK_API_KEY."
    payload = {
        "model": os.environ.get("DEEPSEEK_VISION_MODEL", "deepseek-flash"),
        "max_tokens": 1024,
        "thinking": {"type": "disabled"},
        "system": _dated_prompt(_VISUAL_PROMPT),
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": q},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image}},
                ],
            }
        ],
        "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}],
    }
    data: Any = None
    for attempt in range(3):
        try:
            response = requests.post(
                _DEEPSEEK_URL,
                headers={"x-api-key": api_key, "anthropic-version": _ANTHROPIC_VERSION},
                json=payload,
                timeout=90,
            )
            response.raise_for_status()
            data = response.json()
            break
        except requests.ConnectionError as exc:
            if attempt == 2:
                return f"visual_web_search request failed after 3 attempts: {_request_error_detail(exc)}"
            time.sleep(1.0 * (2**attempt))
        except requests.RequestException as exc:
            return f"visual_web_search request failed: {_request_error_detail(exc)}"
        except ValueError as exc:
            return f"visual_web_search returned invalid JSON: {exc}"
    if not isinstance(data, dict):
        return "visual_web_search returned invalid JSON."
    if data.get("error"):
        return f"visual_web_search failed: {data['error']}"
    text = _anthropic_text(data)
    if not text:
        return "visual_web_search returned no text content."
    return text[:8_000]


def _chat_text(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return ""
    content = (choices[0].get("message") or {}).get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [str(item.get("text", "")).strip() for item in content if isinstance(item, dict) and item.get("text")]
        return "\n".join(part for part in parts if part)
    return ""


def deepseek_vlm_search(query: str, image: str = "") -> str:
    q = (query or "").strip()
    if not q:
        return "deepseek_vlm_search failed: query is required."
    if not (image or "").strip():
        return "deepseek_vlm_search failed: no video frame or image has been received yet, so there is no image to analyze."
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        return "deepseek_vlm_search failed: set DEEPSEEK_API_KEY."
    payload = {
        "model": "deepseek-v4-flash-vision-exp",
        "thinking": {"type": "disabled"},
        "messages": [
            {"role": "system", "content": _dated_prompt(_VLM_PROMPT)},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": q},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}", "detail": "high"}},
                ],
            },
        ],
        "max_tokens": 1024,
    }
    data: Any = None
    for attempt in range(3):
        try:
            response = requests.post(
                _DEEPSEEK_CHAT_URL,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=90,
            )
            response.raise_for_status()
            data = response.json()
            break
        except requests.ConnectionError as exc:
            if attempt == 2:
                return f"deepseek_vlm_search request failed after 3 attempts: {_request_error_detail(exc)}"
            time.sleep(1.0 * (2**attempt))
        except requests.RequestException as exc:
            return f"deepseek_vlm_search request failed: {_request_error_detail(exc)}"
        except ValueError as exc:
            return f"deepseek_vlm_search returned invalid JSON: {exc}"
    if not isinstance(data, dict):
        return "deepseek_vlm_search returned invalid JSON."
    text = _chat_text(data)
    if not text:
        return "deepseek_vlm_search returned no text content."
    return text[:8_000]


_VISION_FUNCTIONS: dict[str, Callable[..., str]] = {
    "get_current_datetime": get_current_datetime,
    "DeepSeek": ask_deepseek,
    "deepseek_vlm_search": deepseek_vlm_search,
    "get_current_location": get_current_location,
    "get_city_weather": get_city_weather,
    "visual_web_search": visual_web_search,
}
# Per-tool override; everything else uses the executor default.
_VISION_TIMEOUTS = {"deepseek_vlm_search": 60.0, "visual_web_search": 60.0}
# The executor injects the latest frame into these; the model never passes it.
_VISION_FRAME_TOOLS = frozenset({"deepseek_vlm_search", "visual_web_search"})


class VisionToolExecutor:
    """The vision tool set: plain-text outputs, and the latest frame for frame tools."""

    uses_latest_frame = True

    def __init__(self, timeout_s: float = 10.0) -> None:
        self._timeout_s = timeout_s

    @property
    def realtime_tools(self) -> list[dict[str, Any]]:
        return [{"type": "function", **definition["function"]} for definition in _VISION_DEFINITIONS]

    async def run(self, name: str, arguments: str, frame_b64: str = "") -> str:
        """Return the text the model sees: the tool output, or the error string."""
        fn = _VISION_FUNCTIONS.get(name)
        if fn is None:
            return f"tool not found: {name}"
        try:
            kwargs = json.loads(arguments or "{}")
            if not isinstance(kwargs, dict):
                raise ValueError("tool arguments must be a JSON object")
            if name in _VISION_FRAME_TOOLS:
                kwargs["image"] = frame_b64
            output = await asyncio.wait_for(
                asyncio.to_thread(fn, **kwargs), timeout=_VISION_TIMEOUTS.get(name, self._timeout_s)
            )
        except asyncio.TimeoutError:
            return "tool timeout"
        except Exception as exc:
            return str(exc)
        return output
