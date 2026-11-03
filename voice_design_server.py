#!/usr/bin/env python3
"""本地中文多维声音设计台后端。"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import io
import json
import lzma
import math
import os
import shutil
import struct
import subprocess
import tempfile
import threading
import traceback
import unicodedata
import uuid
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


ROOT = Path(__file__).resolve().parent
UI_FILE = ROOT / "voice_design_ui.html"
LAB_FILE = ROOT / "speaker_lab.html"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models" / "huggingface"
SPEAKER_FILE = ROOT / "data" / "chattts_speakers.json"
CUSTOM_FILE = ROOT / "data" / "custom_speakers.json"
EMBED_DIM = 768
SAMPLE_RATE = 24000
MAX_UPLOAD_BYTES = 40 * 1024 * 1024
MAX_PROMPT_SECONDS = 12
LZMA_RAW_FILTERS = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}]
VOICE_COUNT = 60
VOICE_CARDS = [
    {"id": f"speaker_{index:03d}", "name": f"普通话音色 {index:02d}"}
    for index in range(1, VOICE_COUNT + 1)
]
FIXED_NAMES = {card["id"]: card["name"] for card in VOICE_CARDS}
_custom_lock = threading.Lock()


# ---------------------------------------------------------------- voice codec
# ChatTTS 音色串格式（2Noise/ChatTTS model/speaker.py）：
#   embedding: float16(768) 二进制 -> LZMA(RAW, LZMA2 preset 9e) -> base16384
#   spk_smp:   uint16 shape 头(4 字节) + uint16 token 数组 -> 同上压缩编码


def _b14():
    try:
        import pybase16384
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "缺少 pybase16384 依赖；请在项目 .venv 中安装 requirements.txt 后重试。"
        ) from exc
    return pybase16384


def emb_decode(payload: str) -> list[float]:
    """把音色嵌入串还原成 768 个数值。"""
    try:
        raw = lzma.decompress(
            _b14().decode_from_string(payload),
            format=lzma.FORMAT_RAW,
            filters=LZMA_RAW_FILTERS,
        )
        if len(raw) % 2:
            raise ValueError("字节数不是 2 的倍数")
        count = len(raw) // 2
        values = list(struct.unpack(f"<{count}e", raw))
    except (lzma.LZMAError, ValueError, struct.error) as exc:
        raise RuntimeError(f"音色嵌入串无法解码：{exc}") from exc
    return values


def emb_encode(values: list[float]) -> str:
    """数值列表 -> 音色嵌入串（与官方 sample_random_speaker 输出同格式）。"""
    packed = struct.pack(f"<{len(values)}e", *[float(v) for v in values])
    return _b14().encode_to_string(
        lzma.compress(packed, format=lzma.FORMAT_RAW, filters=LZMA_RAW_FILTERS)
    )


def smp_decode(payload: str) -> tuple[list[int], list[int]]:
    """解码音频提示串（spk_smp）：返回 (shape, uint16 数值)。"""
    try:
        raw = _b14().decode_from_string(payload)
        shape = list(struct.unpack("<2H", raw[:4]))
        body = lzma.decompress(raw[4:], format=lzma.FORMAT_RAW, filters=LZMA_RAW_FILTERS)
        if len(body) % 2:
            raise ValueError("字节数不是 2 的倍数")
        values = list(struct.unpack(f"<{len(body) // 2}H", body))
    except (lzma.LZMAError, ValueError, struct.error) as exc:
        raise RuntimeError(f"音频提示串无法解码：{exc}") from exc
    return shape, values


def smp_encode(shape: list[int], values: list[int]) -> str:
    head = struct.pack("<2H", int(shape[0]) & 0xFFFF, int(shape[1]) & 0xFFFF)
    body = struct.pack(f"<{len(values)}H", *[int(v) & 0xFFFF for v in values])
    return _b14().encode_to_string(
        head + lzma.compress(body, format=lzma.FORMAT_RAW, filters=LZMA_RAW_FILTERS)
    )


def _now_text() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _new_custom_id() -> str:
    return f"custom_{dt.datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}"


def _valid_custom(record: object) -> bool:
    return (
        isinstance(record, dict)
        and bool(record.get("id"))
        and bool(record.get("name"))
        and record.get("type") in ("emb", "smp")
        and bool(record.get("embedding") or record.get("spk_smp"))
    )


def _read_custom_file() -> list[dict[str, object]]:
    if not CUSTOM_FILE.exists():
        return []
    try:
        payload = json.loads(CUSTOM_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    voices = payload.get("voices", [])
    if not isinstance(voices, list):
        return []
    return [dict(item) for item in voices if _valid_custom(item)]


def custom_voices_meta() -> list[dict[str, object]]:
    with _custom_lock:
        voices = _read_custom_file()
    return [
        {
            "id": item["id"],
            "name": item["name"],
            "type": item["type"],
            "source": item.get("source", ""),
            "created": item.get("created", ""),
        }
        for item in voices
    ]


def custom_voice_by_id(voice_id: str) -> dict[str, object] | None:
    with _custom_lock:
        for item in _read_custom_file():
            if str(item["id"]) == voice_id:
                return item
    return None


def add_custom_voice(record: dict[str, object]) -> dict[str, object]:
    with _custom_lock:
        voices = _read_custom_file()
        voices.append(record)
        CUSTOM_FILE.parent.mkdir(parents=True, exist_ok=True)
        CUSTOM_FILE.write_text(
            json.dumps({"format": 1, "voices": voices}, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
    return record


def mutate_custom_voice(voice_id: str, **fields: object) -> bool:
    with _custom_lock:
        voices = _read_custom_file()
        found = False
        for item in voices:
            if str(item["id"]) == voice_id:
                item.update({key: value for key, value in fields.items() if value is not None})
                found = True
                break
        if not found:
            return False
        CUSTOM_FILE.parent.mkdir(parents=True, exist_ok=True)
        CUSTOM_FILE.write_text(
            json.dumps({"format": 1, "voices": voices}, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        return True


def delete_custom_voice(voice_id: str) -> bool:
    with _custom_lock:
        voices = _read_custom_file()
        remaining = [item for item in voices if str(item["id"]) != voice_id]
        if len(remaining) == len(voices):
            return False
        CUSTOM_FILE.parent.mkdir(parents=True, exist_ok=True)
        CUSTOM_FILE.write_text(
            json.dumps({"format": 1, "voices": remaining}, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        return True


def _read_fixed_embeddings() -> dict[str, str]:
    if not SPEAKER_FILE.exists():
        return {}
    try:
        payload = json.loads(SPEAKER_FILE.read_text(encoding="utf-8"))
        speakers = payload.get("speakers", [])
    except (OSError, ValueError, AttributeError):
        return {}
    if not isinstance(speakers, list):
        return {}
    return {
        str(item.get("id")): str(item.get("embedding"))
        for item in speakers
        if isinstance(item, dict) and item.get("id") and item.get("embedding")
    }


def fixed_embedding(voice_id: str) -> str:
    table = _read_fixed_embeddings()
    if voice_id not in table:
        raise RuntimeError(
            "固定音色数据尚未初始化；请先在“设计台”页面完成一次生成，或重启服务后再试。"
        )
    return table[voice_id]


def resolve_speaker(voice: object) -> tuple[dict[str, str | None], str]:
    """把前端音色描述解析成 ChatTTS 推理所需的 (参数, 标签)。"""
    if not isinstance(voice, dict):
        raise RuntimeError("音色参数格式错误。")
    kind = str(voice.get("kind", ""))
    if kind == "fixed":
        voice_id = str(voice.get("id", ""))
        return {"spk_emb": fixed_embedding(voice_id), "spk_smp": None, "txt_smp": None}, voice_id
    if kind == "custom":
        record = custom_voice_by_id(str(voice.get("id", "")))
        if record is None:
            raise RuntimeError("找不到该自定义音色，可能已被删除。")
        if record["type"] == "smp":
            return (
                {
                    "spk_emb": None,
                    "spk_smp": str(record.get("spk_smp", "")),
                    "txt_smp": str(record.get("txt_smp", "")),
                },
                str(record["id"]),
            )
        return {"spk_emb": str(record.get("embedding", "")), "spk_smp": None, "txt_smp": None}, str(record["id"])
    if kind == "vector":
        values = voice.get("vector")
        if not isinstance(values, list) or len(values) != EMBED_DIM:
            raise RuntimeError(f"编辑器数值必须是 {EMBED_DIM} 个。")
        checked: list[float] = []
        for item in values:
            try:
                number = float(item)
            except (TypeError, ValueError) as exc:
                raise RuntimeError("编辑器数值中包含非数字。") from exc
            if not math.isfinite(number):
                raise RuntimeError("编辑器数值中包含无穷或 NaN。")
            checked.append(max(-1000.0, min(1000.0, number)))
        return {"spk_emb": emb_encode(checked), "spk_smp": None, "txt_smp": None}, "editor"
    if kind == "emb_string":
        payload = str(voice.get("string", ""))
        values = emb_decode(payload)
        if len(values) != EMBED_DIM:
            raise RuntimeError(f"嵌入串维度为 {len(values)}，应为 {EMBED_DIM}，无法使用。")
        return {"spk_emb": payload, "spk_smp": None, "txt_smp": None}, "imported"
    if kind == "smp_string":
        payload = str(voice.get("string", ""))
        smp_decode(payload)
        return (
            {"spk_emb": None, "spk_smp": payload, "txt_smp": str(voice.get("text", ""))},
            "imported",
        )
    raise RuntimeError("未知音色类型。")


def clamp(value: object, low: float, high: float, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


def esc(value: float) -> str:
    return f"{value:.5f}"


def run(command: list[str]) -> None:
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        detail = (completed.stderr or completed.stdout)[-4000:]
        raise RuntimeError(detail)


# ---------------------------------------------------------------- pinyin input
# ChatTTS 的输入层是 BERT 中文字符 token（词表 21128，没有拼音/音素 token），
# 因此“拼音输入”通过同音字映射实现：每个带调音节换成 assets/pinyin_syllables.json
# 里的首选同音字，再交给模型逐字发音。映射表由 build_pinyin_map.py 用 pypinyin
# 反查生成（候选字限定在模型词表内，优先“主读音匹配”和 ChatTTS 官方验证过读音的
# 字）。合成时跳过 refine_text，避免模型把生造音节串改写成正常词语。

PINYIN_FILE = ROOT / "assets" / "pinyin_syllables.json"
ERHUA_CHAR = "儿"
PINYIN_MAX_SYLLABLES = 400
PINYIN_MAX_CHARS = 2400
_PINYIN_PUNCT = "，,、。.；;：:！!？?…—-·~～"
_TONE_MARKS = {"\u0304": "1", "\u0301": "2", "\u030c": "3", "\u0300": "4"}
_pinyin_lock = threading.Lock()
_pinyin_loaded = False
_pinyin_syllables: dict[str, list[str]] = {}
_pinyin_default_tone: dict[str, str] = {}
_pinyin_bases: frozenset[str] = frozenset()


def _pinyin_load() -> None:
    global _pinyin_loaded, _pinyin_syllables, _pinyin_default_tone, _pinyin_bases
    with _pinyin_lock:
        if _pinyin_loaded:
            return
        if not PINYIN_FILE.exists():
            raise RuntimeError(
                "缺少拼音映射表 assets/pinyin_syllables.json；"
                "请在项目目录运行 uv run --no-project --with pypinyin build_pinyin_map.py 生成。"
            )
        payload = json.loads(PINYIN_FILE.read_text(encoding="utf-8"))
        syllables = payload.get("syllables")
        defaults = payload.get("default_tone")
        if not isinstance(syllables, dict) or not isinstance(defaults, dict) or not syllables:
            raise RuntimeError("拼音映射表内容不完整，请重新生成。")
        _pinyin_syllables = {str(k): [str(c) for c in v] for k, v in syllables.items()}
        _pinyin_default_tone = {str(k): str(v) for k, v in defaults.items()}
        _pinyin_bases = frozenset(key[:-1] for key in _pinyin_syllables)
        _pinyin_loaded = True


def _decode_pinyin_chunk(chunk: str) -> tuple[str, dict[int, str], dict[int, str]]:
    """把一个连续段解码成 (小写字母串, 带调符号位置->声调, 数字声调的音节末位置->声调)。

    支持三种写法：声调数字（ni3）、Unicode 声调符号（nǐ）、ü 写作 v/u:/ü。
    数字声调同时充当强制音节边界（ni3hao3 -> ni|hao）。
    """
    plains: list[str] = []
    marks: dict[int, str] = {}
    digit_tones: dict[int, str] = {}
    for raw in chunk:
        if raw.isdigit():
            end = len(plains)
            if end == 0:
                raise RuntimeError(f"声调数字“{raw}”前面没有音节字母。")
            if end in digit_tones:
                raise RuntimeError(f"音节“{''.join(plains)}”的结尾写了多个声调数字。")
            digit = "5" if raw == "0" else raw
            if digit not in "12345":
                raise RuntimeError(f"声调数字只能是 1-5（轻声可写 5 或 0）。")
            digit_tones[end] = digit
            continue
        base_letters: list[str] = []
        tone = ""
        for ch in unicodedata.normalize("NFD", raw):
            if ch in _TONE_MARKS:
                tone = _TONE_MARKS[ch]
            elif ch == "\u0308":  # ü 的分音符
                if base_letters and base_letters[-1] == "u":
                    base_letters[-1] = "v"
            elif ch.isascii() and ch.isalpha():
                base_letters.append(ch.lower())
            else:
                raise RuntimeError(f"拼音里包含无法识别的字符“{raw}”。")
        if not base_letters:
            raise RuntimeError(f"拼音里包含无法识别的字符“{raw}”。")
        if tone:
            marks[len(plains)] = tone
        plains.extend(base_letters)
    return "".join(plains), marks, digit_tones


def _segment_pinyin(
    plains: str, marks: dict[int, str], digit_tones: dict[int, str]
) -> list[tuple[str, str, bool]]:
    """把字母串切成音节。返回 (基础音节, 声调或空, 是否儿化后缀) 列表。

    规则：最长匹配优先（xian 优先于 xi+an）；数字声调位置是强制边界；
    声调符号归属其所在音节；普通切分失败时尝试“音节+r”儿化（huar -> hua+儿）。
    """
    total = len(plains)
    forced = sorted(digit_tones)

    def next_boundary(start: int) -> int:
        for pos in forced:
            if pos > start:
                return pos
        return total

    def tone_of(start: int, end: int) -> str:
        inner = {marks[pos] for pos in marks if start <= pos < end}
        if len(inner) > 1:
            raise RuntimeError(f"“{plains[start:end]}”里出现了两种不同的声调符号。")
        tone = next(iter(inner), "")
        digit = digit_tones.get(end, "")
        if tone and digit and tone != digit:
            raise RuntimeError(
                f"音节“{plains[start:end]}”的声调符号（{tone}）和数字（{digit}）不一致。"
            )
        return digit or tone

    failed: set[int] = set()

    def recurse(start: int) -> list[tuple[str, str, bool]] | None:
        if start == total:
            return []
        if start in failed:
            return None
        limit = next_boundary(start)
        span = plains[start:limit]
        # 儿化解释优先：剩余段恰好是 基础音节 + "r"（huar -> hua+儿）。
        # er 本身是音节，不适用；儿化必须优于普通切分，否则 huar 会被切成 hu+a+儿。
        if (
            len(span) >= 2
            and span[-1] == "r"
            and span not in _pinyin_bases
            and span[:-1] in _pinyin_bases
        ):
            inner = {marks[pos] for pos in marks if start <= pos < limit - 1}
            if len(inner) > 1:
                raise RuntimeError(f"“{span}”里出现了两种不同的声调符号。")
            mark_tone = next(iter(inner), "")
            digit_tone = digit_tones.get(limit, "")
            if mark_tone and digit_tone and mark_tone != digit_tone:
                raise RuntimeError(
                    f"音节“{span[:-1]}”的声调符号（{mark_tone}）和数字（{digit_tone}）不一致。"
                )
            rest = recurse(limit)
            if rest is not None:
                return [(span[:-1], digit_tone or mark_tone, False), ("r", "", True), *rest]
        for length in range(min(6, limit - start), 0, -1):
            end = start + length
            base = plains[start:end]
            if base not in _pinyin_bases:
                continue
            try:
                tone = tone_of(start, end)
            except RuntimeError:
                raise  # 声调冲突无论怎么切都算输入错误
            rest = recurse(end)
            if rest is not None:
                return [(base, tone, False), *rest]
        failed.add(start)
        return None

    items = recurse(0)
    if items is None:
        raise RuntimeError(f"“{plains}”无法切分成有效音节；请在音节之间加空格，或检查拼写。")
    return items


def _resolve_syllable(base: str, tone: str) -> tuple[str, str]:
    """确定最终声调并取首选同音字。未标声调时用映射表的默认声调（近似最常见读音）。"""
    if not tone:
        tone = _pinyin_default_tone.get(base, "")
    entry = _pinyin_syllables.get(base + tone) if tone else None
    if not entry:
        available = [t for t in "12345" if (base + t) in _pinyin_syllables]
        hint = (
            f"（“{base}”可用的声调：{'、'.join(available)}）" if available else "，请检查拼写。"
        )
        if tone:
            raise RuntimeError(f"音节“{base}”没有声调 {tone} 的对应汉字{hint}")
        raise RuntimeError(f"音节“{base}”没有任何对应汉字{hint}")
    return tone, entry[0]


def convert_pinyin(text: str) -> tuple[str, list[dict[str, object]], list[str]]:
    """把拼音文本转换成同音汉字串。返回 (汉字串, 音节信息列表, 错误列表)。"""
    _pinyin_load()
    raw = str(text).strip()
    if not raw:
        raise RuntimeError("请输入拼音文本。")
    chars: list[str] = []
    info: list[dict[str, object]] = []
    errors: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if not buffer:
            return
        chunk = "".join(buffer)
        buffer.clear()
        try:
            plains, marks, digit_tones = _decode_pinyin_chunk(chunk)
            for base, tone, erhua in _segment_pinyin(plains, marks, digit_tones):
                if erhua:
                    chars.append(ERHUA_CHAR)
                    info.append(
                        {"syl": "r", "char": ERHUA_CHAR, "tone": "", "erhua": True, "explicit": True}
                    )
                    continue
                resolved, char = _resolve_syllable(base, tone)
                chars.append(char)
                info.append(
                    {
                        "syl": base + resolved,
                        "char": char,
                        "tone": resolved,
                        "erhua": False,
                        "explicit": bool(tone),
                    }
                )
        except RuntimeError as exc:
            errors.append(str(exc))

    for ch in raw:
        if ch.isspace() or ch in "'’‘`":
            flush()
        elif ch in _PINYIN_PUNCT:
            flush()
            chars.append(ch)
        else:
            buffer.append(ch)
    flush()

    if not info and not errors:
        errors.append("没有识别到任何拼音音节。")
    if len(info) > PINYIN_MAX_SYLLABLES:
        errors.append(f"音节太多（{len(info)} 个），请控制在 {PINYIN_MAX_SYLLABLES} 个以内。")
    if len("".join(chars)) > PINYIN_MAX_CHARS:
        errors.append(f"文本太长，请控制在 {PINYIN_MAX_CHARS} 个汉字以内。")
    return "".join(chars), info, errors


def pinyin_preview(payload: dict[str, object]) -> dict[str, object]:
    """供前端实时预览：显示每个音节选用的同音字和错误。"""
    text = str(payload.get("text", ""))[: PINYIN_MAX_CHARS * 2]
    chars, info, errors = convert_pinyin(text)
    return {"chars": chars, "syllables": info, "errors": errors[:8]}


class ChatTTSEngine:
    """Lazily load ChatTTS and persist a stable set of speaker embeddings."""

    def __init__(self) -> None:
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._chat = None
        self._chattts = None
        self._torch = None
        self._numpy = None
        self._speakers: dict[str, str] = {}
        self._state = "未加载"
        self._error = ""

    def status(self) -> dict[str, str]:
        state = self._state
        if self._chat is None and not self._error and importlib.util.find_spec("ChatTTS") is None:
            state = "未安装 ChatTTS"
        return {"state": state, "error": self._error}

    def _load(self) -> None:
        if self._chat is not None:
            return
        with self._load_lock:
            if self._chat is not None:
                return
            # A failed download is usually transient. Do not permanently lock the
            # engine in the failed state; the next Generate click retries/resumes
            # the Hugging Face download from the existing cache.
            self._error = ""
            self._state = "正在下载或加载模型"
            try:
                # Keep the model cache inside the project so it is easy to find and move.
                MODEL_DIR.mkdir(parents=True, exist_ok=True)
                os.environ["HF_HOME"] = str(MODEL_DIR)
                os.environ["HF_HUB_CACHE"] = str(MODEL_DIR / "hub")
                os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
                import ChatTTS
                import numpy as np
                import torch
                import huggingface_hub.constants as hf_constants

                device = None
                if torch.cuda.is_available():
                    device = torch.device("cuda")
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = torch.device("mps")

                endpoints = [
                    endpoint.strip().rstrip("/")
                    for endpoint in os.environ.get(
                        "VOICE_MODEL_ENDPOINTS",
                        "https://huggingface.co,https://hf-mirror.com",
                    ).split(",")
                    if endpoint.strip()
                ]
                load_errors: list[str] = []
                chat = None
                loaded = False
                for endpoint in dict.fromkeys(endpoints):
                    os.environ["HF_ENDPOINT"] = endpoint
                    # huggingface_hub reads HF_ENDPOINT at import time. ChatTTS
                    # imports snapshot_download before this loop, so update the
                    # already-loaded constants and URL template for each retry.
                    hf_constants.ENDPOINT = endpoint
                    hf_constants.HUGGINGFACE_CO_URL_TEMPLATE = (
                        endpoint + "/{repo_id}/resolve/{revision}/{filename}"
                    )
                    try:
                        candidate = ChatTTS.Chat()
                        loaded = candidate.load(source="huggingface", device=device, compile=False)
                    except Exception as exc:
                        load_errors.append(f"{endpoint}: {exc}")
                        continue
                    if loaded:
                        chat = candidate
                        break
                    load_errors.append(f"{endpoint}: ChatTTS 未返回成功状态")
                if not loaded or chat is None:
                    details = "；".join(load_errors[-len(endpoints):])
                    raise RuntimeError(
                        "模型下载失败。已依次尝试："
                        + ", ".join(endpoints)
                        + (f"。详情：{details}" if details else "。")
                    )
                self._chattts = ChatTTS
                self._torch = torch
                self._numpy = np
                self._chat = chat
                self._load_speakers()
                self._state = "模型已就绪"
            except Exception as exc:
                self._state = "加载失败"
                self._error = (
                    f"ChatTTS 初始化失败：{exc}\n"
                    "请确认已用 Python 3.11 安装 requirements.txt，网络可访问模型仓库，"
                    "并留出足够的内存和磁盘空间。"
                )
                raise RuntimeError(self._error) from exc

    def _load_speakers(self) -> None:
        saved: list[dict[str, object]] = []
        if SPEAKER_FILE.exists():
            try:
                payload = json.loads(SPEAKER_FILE.read_text(encoding="utf-8"))
                saved = payload.get("speakers", [])
            except (OSError, ValueError, AttributeError):
                saved = []

        by_id = {
            str(item.get("id")): str(item.get("embedding"))
            for item in saved
            if isinstance(item, dict) and item.get("id") and item.get("embedding")
        }
        if len(by_id) < VOICE_COUNT:
            self._state = "正在准备 60 个固定音色"
            torch = self._torch
            assert torch is not None and self._chat is not None
            for index in range(1, VOICE_COUNT + 1):
                voice_id = f"speaker_{index:03d}"
                if voice_id in by_id:
                    continue
                torch.manual_seed(2_408_170_68 + index)
                by_id[voice_id] = self._chat.sample_random_speaker()

            SPEAKER_FILE.parent.mkdir(parents=True, exist_ok=True)
            SPEAKER_FILE.write_text(
                json.dumps(
                    {
                        "format": 1,
                        "model": "2Noise/ChatTTS",
                        "speakers": [
                            {
                                "id": card["id"],
                                "name": card["name"],
                                "embedding": by_id[card["id"]],
                            }
                            for card in VOICE_CARDS
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        self._speakers = {card["id"]: by_id[card["id"]] for card in VOICE_CARDS}

    def synthesize(self, text: str, rate: int, voice: object, mode: str = "zh"):
        # 拼音模式先转换（不加载模型即可报出拼写错误），再合成同音汉字串。
        converted = ""
        if mode == "pinyin":
            converted, _info, errors = convert_pinyin(text)
            if errors:
                detail = "；".join(str(e).rstrip("。") for e in errors[:3])
                raise RuntimeError(detail + ("等。" if len(errors) > 3 else "。"))
        self._load()
        speaker, label = resolve_speaker(voice)

        # ChatTTS has shared model state. Serialize inference to avoid concurrent
        # requests corrupting its random-generator or decoder state.
        with self._infer_lock:
            assert self._chat is not None and self._chattts is not None
            assert self._numpy is not None
            speed = int(clamp(rate, 1, 9, 5))
            final_text = converted if mode == "pinyin" else text
            seed_text = f"{label}\0{mode}\0{final_text}\0{rate}".encode("utf-8")
            seed = int(hashlib.sha256(seed_text).hexdigest()[:8], 16)
            infer_params = self._chattts.Chat.InferCodeParams(
                spk_emb=speaker["spk_emb"],
                spk_smp=speaker["spk_smp"],
                txt_smp=speaker["txt_smp"] or None,
                prompt=f"[speed_{speed}]",
                manual_seed=seed,
                temperature=0.3,
                top_P=0.7,
                top_K=20,
                show_tqdm=False,
            )
            try:
                audio_list = self._chat.infer(
                    final_text,
                    params_infer_code=infer_params,
                    lang="zh",
                    # 拼音模式必须跳过 refine_text，否则模型可能把生造音节串
                    # 改写成读音相近的正常词语，失去逐音节控制。
                    skip_refine_text=(mode == "pinyin"),
                    do_text_normalization=True,
                    do_homophone_replacement=True,
                )
            except ValueError as exc:
                # ChatTTS 在“音频提示音色 + 极短文本”时可能一段音频都生成不了
                raise RuntimeError(
                    "ChatTTS 没能生成音频；使用提取音色时请换一段更长的文本（建议 10 字以上）重试。"
                ) from exc
            if not audio_list:
                raise RuntimeError("ChatTTS 没有返回音频，请缩短文本后重试。")
            samples = self._numpy.asarray(audio_list[0], dtype=self._numpy.float32).reshape(-1)
            if samples.size == 0:
                raise RuntimeError("ChatTTS 返回了空音频，请重试。")
            pcm = self._numpy.clip(samples, -1.0, 1.0)
            pcm = (pcm * 32767.0).astype("<i2", copy=False)
            return pcm.tobytes(), int(getattr(self._chat, "sample_rate", 24000))


ENGINE = ChatTTSEngine()


def render_audio(params: dict[str, object]) -> tuple[bytes, Path]:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("找不到 ffmpeg。请安装 ffmpeg 并加入 PATH。")
    text = str(params.get("chinese", "你好，我想设计一种自然、温暖、清晰的声音。"))[:2400].strip()
    if not text:
        raise RuntimeError("请输入要朗读的文本。")
    text_mode = "pinyin" if str(params.get("text_mode", "zh")) == "pinyin" else "zh"
    voice_id = str(params.get("base_voice", "speaker_001"))
    base_voice = (
        {"kind": "custom", "id": voice_id}
        if voice_id.startswith("custom_")
        else {"kind": "fixed", "id": voice_id}
    )
    rate = int(clamp(params.get("rate"), 1, 9, 5))
    pitch = clamp(params.get("pitch"), -8, 8, 0)
    pitch_variation = clamp(params.get("pitch_variation"), 0, 100, 0)
    brightness = clamp(params.get("brightness"), -8, 8, 0)
    warmth = clamp(params.get("warmth"), -8, 8, 0)
    body = clamp(params.get("body"), -8, 8, 0)
    nasality = clamp(params.get("nasality"), 0, 8, 0)
    tension = clamp(params.get("tension"), 0, 8, 0)
    clarity = clamp(params.get("clarity"), 0, 8, 0)
    presence = clamp(params.get("presence"), 0, 8, 0)
    air = clamp(params.get("air"), 0, 8, 0)
    sibilance = clamp(params.get("sibilance"), -8, 8, 0)
    breathiness = clamp(params.get("breathiness"), 0, 8, 0)
    roughness = clamp(params.get("roughness"), 0, 8, 0)
    smoothness = clamp(params.get("smoothness"), 0, 8, 0)
    compression = clamp(params.get("compression"), 0, 8, 0)
    distance = clamp(params.get("distance"), 0, 100, 0)
    room = clamp(params.get("room"), 0, 100, 0)
    volume = clamp(params.get("volume"), -8, 8, 0)

    ratio = 2 ** (pitch / 12.0)
    lowpass = max(4500, min(16000, 14000 - smoothness * 900 - distance * 35))
    highpass = max(40, min(500, 55 + distance * 0.7))
    filters = [
        "aresample=44100",
        f"asetrate=44100*{esc(ratio)},aresample=44100,atempo={esc(1 / ratio)}",
        f"highpass=f={esc(highpass)}",
        f"lowpass=f={esc(lowpass)}",
        f"equalizer=f=110:t=q:w=0.75:g={esc(body)}",
        f"equalizer=f=250:t=q:w=0.8:g={esc(warmth)}",
        f"equalizer=f=1800:t=q:w=0.9:g={esc(brightness)}",
        f"equalizer=f=1200:t=q:w=1:g={esc(nasality)}",
        f"equalizer=f=2200:t=q:w=1:g={esc(tension)}",
        f"equalizer=f=3200:t=q:w=1:g={esc(clarity)}",
        f"equalizer=f=4800:t=q:w=1:g={esc(presence)}",
        f"equalizer=f=7000:t=q:w=0.8:g={esc(air + sibilance)}",
    ]
    if pitch_variation > 0.1:
        filters.append(f"vibrato=f=5:d={esc(pitch_variation / 100)}")
    if roughness > 0.1:
        bits = max(5, 16 - roughness * 1.2)
        filters.append(f"acrusher=bits={esc(bits)}:mix={esc(roughness / 40)}")
    if compression > 0.1:
        filters.append(f"acompressor=threshold=-20dB:ratio={esc(1 + compression * 0.8)}:attack=20:release=140")
    filters.append(f"volume={esc(10 ** (volume / 20))}")
    chain = ",".join(filters)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"voice_{dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex[:6]}.wav"
    saved_path = OUTPUT_DIR / filename
    with tempfile.TemporaryDirectory(prefix="voice-design-server-") as temp:
        source_path, render_path = Path(temp) / "source.wav", Path(temp) / "render.wav"
        pcm, sample_rate = ENGINE.synthesize(text, rate, base_voice, mode=text_mode)
        with wave.open(str(source_path), "wb") as source_file:
            source_file.setnchannels(1)
            source_file.setsampwidth(2)
            source_file.setframerate(sample_rate)
            source_file.writeframes(pcm)
        graph = [f"[0:a]{chain}[voice]"]
        final = "[voice]"
        if room > 0.1:
            delay, decay = int(30 + room * 2.5), 0.18 + room / 400
            graph.append(f"[voice]aecho=0.8:0.88:{delay}:{esc(decay)}[room]")
            final = "[room]"
        if breathiness > 0.1:
            noise_level = 0.002 + breathiness * 0.0015
            graph.append(f"anoisesrc=c=white:r=44100:a={esc(noise_level)}:d=60[noise]")
            graph.append("[noise]highpass=f=3500,lowpass=f=11000[airnoise]")
            graph.append(f"{final}[airnoise]amix=inputs=2:duration=first:weights=1 0.35[breathy]")
            final = "[breathy]"
        graph.append(f"{final}loudnorm=I=-16:TP=-1.5:LRA=11[out]")
        run([
            "ffmpeg", "-y", "-i", str(source_path), "-filter_complex", ";".join(graph),
            "-map", "[out]", "-ar", "44100", "-ac", "1", str(render_path),
        ])
        data = render_path.read_bytes()
    saved_path.write_bytes(data)
    return data, saved_path


def wrap_wav(pcm: bytes, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return buffer.getvalue()


def preview_audio(payload: dict[str, object]) -> bytes:
    """用指定音色（固定/自定义/编辑器向量）直接合成试听音频，不经 ffmpeg。"""
    default_text = "你好，这是当前音色的试听。"
    text = str(payload.get("text", default_text)).strip()[:600]
    if not text:
        text = default_text
    text_mode = "pinyin" if str(payload.get("text_mode", "zh")) == "pinyin" else "zh"
    rate = int(clamp(payload.get("rate"), 1, 9, 5))
    voice = payload.get("voice")
    pcm, sample_rate = ENGINE.synthesize(text, rate, voice, mode=text_mode)
    return wrap_wav(pcm, sample_rate)


def speaker_detail(voice_id: str, kind: str) -> dict[str, object]:
    """返回音色的数值数据：emb 为 768 个数值；smp 为分层 token。"""
    if kind == "custom":
        record = custom_voice_by_id(voice_id)
        if record is None:
            raise RuntimeError("找不到该自定义音色，可能已被删除。")
        if record["type"] == "smp":
            shape, values = smp_decode(str(record.get("spk_smp", "")))
            frames = shape[-1] if len(shape) >= 2 else len(values) // max(1, shape[0])
            layers = shape[0] if shape else 1
            rows = [
                values[i * frames : (i + 1) * frames] if frames > 0 else []
                for i in range(layers)
            ]
            return {
                "id": record["id"],
                "kind": "custom",
                "name": record["name"],
                "type": "smp",
                "shape": shape,
                "layers": layers,
                "frames": frames,
                "rows": rows,
                "txt_smp": str(record.get("txt_smp", "")),
                "created": record.get("created", ""),
            }
        vector = emb_decode(str(record.get("embedding", "")))
        return {
            "id": record["id"],
            "kind": "custom",
            "name": record["name"],
            "type": "emb",
            "dim": len(vector),
            "vector": vector,
            "raw": str(record.get("embedding", "")),
            "created": record.get("created", ""),
        }

    embedding = fixed_embedding(voice_id)
    vector = emb_decode(embedding)
    return {
        "id": voice_id,
        "kind": "fixed",
        "name": FIXED_NAMES.get(voice_id, voice_id),
        "type": "emb",
        "dim": len(vector),
        "vector": vector,
        "raw": embedding,
    }


def save_speaker(payload: dict[str, object]) -> dict[str, object]:
    """把编辑器向量或导入的音色串保存为新的自定义音色。"""
    name = str(payload.get("name", "")).strip()[:40]
    mode = str(payload.get("mode", ""))
    source = str(payload.get("source", "editor"))[:60]
    record_id = _new_custom_id()
    created = _now_text()
    if mode == "vector":
        if not name:
            name = "编辑音色"
        values = payload.get("vector")
        if not isinstance(values, list) or len(values) != EMBED_DIM:
            raise RuntimeError(f"编辑器数值必须是 {EMBED_DIM} 个。")
        checked: list[float] = []
        for item in values:
            try:
                number = float(item)
            except (TypeError, ValueError) as exc:
                raise RuntimeError("编辑器数值中包含非数字。") from exc
            if not math.isfinite(number):
                raise RuntimeError("编辑器数值中包含无穷或 NaN。")
            checked.append(max(-1000.0, min(1000.0, number)))
        record = {
            "id": record_id,
            "name": name,
            "type": "emb",
            "embedding": emb_encode(checked),
            "source": source,
            "created": created,
        }
    elif mode == "string":
        raw = str(payload.get("string", ""))
        if not raw:
            raise RuntimeError("请粘贴要导入的音色串。")
        voice_type = str(payload.get("voice_type", "emb"))
        if voice_type == "smp":
            shape, _values = smp_decode(raw)
            if not shape or len(shape) < 2 or shape[0] < 1 or shape[1] < 1:
                raise RuntimeError("音频提示串的形状信息不合法。")
            record = {
                "id": record_id,
                "name": name or "导入提取音色",
                "type": "smp",
                "spk_smp": raw,
                "txt_smp": str(payload.get("text", "")).strip()[:2000],
                "source": "import",
                "created": created,
            }
        else:
            values = emb_decode(raw)
            if len(values) != EMBED_DIM:
                raise RuntimeError(f"嵌入串维度为 {len(values)}，应为 {EMBED_DIM}，无法导入。")
            record = {
                "id": record_id,
                "name": name or "导入音色",
                "type": "emb",
                "embedding": raw,
                "source": "import",
                "created": created,
            }
    else:
        raise RuntimeError("未知的保存方式。")
    add_custom_voice(record)
    return {
        "voice": {
            "id": record["id"],
            "name": record["name"],
            "type": record["type"],
            "created": created,
        }
    }


def update_speaker(payload: dict[str, object]) -> dict[str, object]:
    voice_id = str(payload.get("id", ""))
    record = custom_voice_by_id(voice_id)
    if record is None:
        raise RuntimeError("找不到该自定义音色。")
    fields: dict[str, object] = {}
    if "name" in payload:
        name = str(payload.get("name", "")).strip()[:40]
        if not name:
            raise RuntimeError("音色名称不能为空。")
        fields["name"] = name
    if "txt_smp" in payload and record["type"] == "smp":
        fields["txt_smp"] = str(payload.get("txt_smp", "")).strip()[:2000]
    if not fields:
        raise RuntimeError("没有需要修改的内容。")
    if not mutate_custom_voice(voice_id, **fields):
        raise RuntimeError("保存修改失败。")
    return {"ok": True}


def delete_speaker(payload: dict[str, object]) -> dict[str, object]:
    voice_id = str(payload.get("id", ""))
    if not voice_id.startswith("custom_"):
        raise RuntimeError("只能删除自定义音色；60 个固定音色不可删除。")
    if not delete_custom_voice(voice_id):
        raise RuntimeError("找不到该自定义音色。")
    return {"ok": True}


def load_wav_mono(path: Path) -> "tuple[object, int]":
    """读 WAV 为 mono float32 波形。优先标准库（PCM），失败回退 ffmpeg（支持浮点/非 PCM）。"""
    np = ENGINE._numpy
    assert np is not None
    try:
        with wave.open(str(path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            width = wav_file.getsampwidth()
            rate = wav_file.getframerate()
            if channels < 1 or width not in (1, 2, 3, 4) or rate < 4000:
                raise ValueError("unsupported wav layout")
            frames = wav_file.readframes(wav_file.getnframes())
        block = channels * width
        count = len(frames) // block
        frames = frames[: count * block]
        if width == 1:
            pcm = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
        elif width == 2:
            pcm = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        elif width == 3:
            raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
            values = (
                raw[:, 0].astype(np.int32)
                | (raw[:, 1].astype(np.int32) << 8)
                | (raw[:, 2].astype(np.int32) << 16)
            )
            values = np.where(values & 0x800000, values - 0x1000000, values)
            pcm = values.astype(np.float32) / 8388608.0
        else:
            pcm = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
        if channels > 1:
            pcm = pcm.reshape(-1, channels).mean(axis=1)
        return pcm.reshape(-1), int(rate)
    except wave.Error:
        pass  # 非 PCM WAV，走 ffmpeg
    raw_path = path.with_suffix(".raw")
    run([
        "ffmpeg", "-y", "-i", str(path), "-ac", "1",
        "-ar", str(SAMPLE_RATE), "-f", "s16le", str(raw_path),
    ])
    pcm = np.frombuffer(raw_path.read_bytes(), dtype="<i2").astype(np.float32) / 32768.0
    return pcm.reshape(-1), SAMPLE_RATE


def extract_speaker(wav_bytes: bytes, name: str, text: str) -> dict[str, object]:
    """从上传的 WAV 中提取音频提示（spk_smp）并保存为自定义音色。"""
    if not wav_bytes:
        raise RuntimeError("没有收到音频数据。")
    if len(wav_bytes) > MAX_UPLOAD_BYTES:
        raise RuntimeError("音频文件过大（超过 40MB）。")
    ENGINE._load()
    chat = ENGINE._chat
    np = ENGINE._numpy
    torch = ENGINE._torch
    assert chat is not None and np is not None and torch is not None

    with tempfile.TemporaryDirectory(prefix="voice-extract-") as temp:
        path = Path(temp) / "upload.wav"
        path.write_bytes(wav_bytes)
        try:
            wav, sr = load_wav_mono(path)
        except Exception as exc:
            raise RuntimeError(
                f"无法读取该音频（建议上传 16bit PCM 的 WAV，或安装 ffmpeg 后重试）：{exc}"
            ) from exc
        if sr != SAMPLE_RATE:
            waveform = torch.from_numpy(np.ascontiguousarray(wav)).to(dtype=torch.float32)
            try:
                import torchaudio.functional as ta_f

                waveform = ta_f.resample(waveform, sr, SAMPLE_RATE)
            except Exception:
                scale = SAMPLE_RATE / sr
                size = max(1, int(waveform.numel() * scale))
                waveform = torch.nn.functional.interpolate(
                    waveform.view(1, 1, -1), size=size, mode="linear", align_corners=False
                ).view(-1)
            wav = waveform.cpu().numpy().astype(np.float32)

    if wav.size < SAMPLE_RATE // 2:
        raise RuntimeError("音频太短（不足 0.5 秒），请上传更长的语音。")
    gate = max(1e-4, float(np.abs(wav).max()) * 0.02)
    nonzero = np.nonzero(np.abs(wav) > gate)[0]
    if nonzero.size == 0:
        raise RuntimeError("音频几乎全是静音，无法提取音色。")
    margin = int(SAMPLE_RATE * 0.15)
    start = max(0, int(nonzero[0]) - margin)
    end = min(wav.size, int(nonzero[-1]) + margin)
    wav = wav[start:end]
    max_len = SAMPLE_RATE * MAX_PROMPT_SECONDS
    if wav.size > max_len:
        wav = wav[:max_len]
    seconds = wav.size / SAMPLE_RATE

    with ENGINE._infer_lock:
        spk_smp = chat.sample_audio_speaker(wav)
    shape, _values = smp_decode(spk_smp)

    record = {
        "id": _new_custom_id(),
        "name": name or "提取音色",
        "type": "smp",
        "spk_smp": spk_smp,
        "txt_smp": text,
        "source": "wav",
        "created": _now_text(),
    }
    add_custom_voice(record)
    return {
        "voice": {
            "id": record["id"],
            "name": record["name"],
            "type": "smp",
            "created": record["created"],
        },
        "seconds": round(seconds, 2),
        "layers": shape[0] if shape else 0,
        "frames": shape[-1] if len(shape) >= 2 else len(_values),
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "VoiceDesign/2.0"

    def _send_json(self, status: int, payload: dict[str, object]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_wav(self, audio: bytes, filename: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Disposition", f"inline; filename={filename}")
        self.send_header("Content-Length", str(len(audio)))
        self.end_headers()
        self.wfile.write(audio)

    def _read_json_body(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > MAX_UPLOAD_BYTES:
            raise RuntimeError("请求体过大。")
        raw = self.rfile.read(length).decode("utf-8")
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise RuntimeError("请求体必须是 JSON 对象。")
        return payload

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            data = UI_FILE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if parsed.path == "/lab":
            if not LAB_FILE.exists():
                self.send_error(404)
                return
            data = LAB_FILE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if parsed.path == "/api/voices":
            self._send_json(
                200,
                {
                    "voices": VOICE_CARDS,
                    "custom": custom_voices_meta(),
                    "engine": ENGINE.status(),
                },
            )
            return
        if parsed.path == "/api/speaker/detail":
            try:
                query = parse_qs(parsed.query)
                voice_id = (query.get("id") or [""])[0]
                kind = (query.get("kind") or ["fixed"])[0]
                if not voice_id:
                    raise RuntimeError("缺少音色 id 参数。")
                if kind not in ("fixed", "custom"):
                    raise RuntimeError("kind 参数必须是 fixed 或 custom。")
                self._send_json(200, speaker_detail(voice_id, kind))
            except Exception as exc:
                self._send_json(400, {"error": str(exc) or "读取音色数据失败。"})
            return
        if parsed.path.startswith("/outputs/"):
            name = Path(unquote(parsed.path).replace("/outputs/", "", 1)).name
            target = (OUTPUT_DIR / name).resolve()
            if target.parent == OUTPUT_DIR.resolve() and target.exists():
                data = target.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        try:
            if route == "/api/render":
                length = int(self.headers.get("Content-Length", "0"))
                params = json.loads(self.rfile.read(length).decode("utf-8"))
                audio, saved_path = render_audio(params)
                relative = saved_path.relative_to(ROOT).as_posix()
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Disposition", f"inline; filename={saved_path.name}")
                self.send_header("X-Voice-File", relative)
                self.send_header("Access-Control-Expose-Headers", "X-Voice-File")
                self.send_header("Content-Length", str(len(audio)))
                self.end_headers()
                self.wfile.write(audio)
                return
            if route == "/api/preview":
                payload = self._read_json_body()
                audio = preview_audio(payload)
                self._send_wav(audio, "preview.wav")
                return
            if route == "/api/pinyin/preview":
                payload = self._read_json_body()
                self._send_json(200, pinyin_preview(payload))
                return
            if route == "/api/speaker/save":
                payload = self._read_json_body()
                self._send_json(200, save_speaker(payload))
                return
            if route == "/api/speaker/update":
                payload = self._read_json_body()
                self._send_json(200, update_speaker(payload))
                return
            if route == "/api/speaker/delete":
                payload = self._read_json_body()
                self._send_json(200, delete_speaker(payload))
                return
            if route == "/api/speaker/extract":
                parsed = urlparse(self.path)
                query = parse_qs(parsed.query)
                name = unquote((query.get("name") or [""])[0]).strip()[:40]
                text = unquote((query.get("text") or [""])[0]).strip()[:2000]
                length = int(self.headers.get("Content-Length", "0"))
                if length > MAX_UPLOAD_BYTES:
                    raise RuntimeError("音频文件过大（超过 40MB）。")
                wav_bytes = self.rfile.read(length)
                self._send_json(200, extract_speaker(wav_bytes, name, text))
                return
            self.send_error(404)
        except Exception as exc:  # pragma: no cover
            print(f"Request failed ({self.path}): {exc}", flush=True)
            traceback.print_exc()
            self._send_json(500, {"error": str(exc) or "请求处理失败，请查看服务窗口日志。"})

    def log_message(self, format: str, *args: object) -> None:
        print(format % args)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not UI_FILE.exists():
        raise SystemExit(f"找不到 {UI_FILE}")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"声音设计台：http://{args.host}:{args.port}")
    print(f"音色实验室：http://{args.host}:{args.port}/lab")
    print(f"普通话固定音色：{VOICE_COUNT} 个；首次生成时自动下载并初始化模型。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    import argparse

    main()
