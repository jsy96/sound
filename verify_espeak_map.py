#!/usr/bin/env python3
"""用真实 espeak-ng 验证「IPA -> espeak 音素代码」映射，生成 assets/espeak_ipa_map.json。

三种验证手段，全部基于真实二进制，不依赖记忆：
1. 候选拼写 round-trip：把候选助记代码喂给 [[...]]，用 --ipa 输出比对基本字符；
2. -X 自省：让某声部朗读包含目标音的真实词语，从输出中提取它实际使用的助记代码，
   再对这些代码做 round-trip 验证；
3. 差分验证（控制符号/声调）：对比加/不加标记的 --ipa 输出或合成 WAV 大小。

每个符号记录「全部可用声部」：{符号: {声部: 代码}}。运行时按输入内容选择
覆盖最全的声部合成（用户的输入通常整体落在某一语言的音系内）。

用法（先安装：winget install eSpeak-NG.eSpeak-NG）：
    .venv/Scripts/python.exe verify_espeak_map.py
"""

from __future__ import annotations

import datetime as dt
import json
import re
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_FILE = ROOT / "assets" / "espeak_ipa_map.json"
TIMEOUT = 10
PROBE_VOICES = ["en-us", "fr", "de", "cmn", "hi", "ar", "ru", "vi", "ja", "ko", "nl", "es", "it", "pl", "en-gb", "cmn-latn-pinyin"]

# ---- 1) 候选拼写（假设，需 round-trip 通过才有效）
CANDIDATES: dict[str, list[str]] = {
    "p": ["p"], "b": ["b"], "t": ["t"], "d": ["d"], "k": ["k"], "ɡ": ["g"],
    "q": ["q"], "ʔ": ["?"], "ʈ": ["t`"], "ɖ": ["d`"],
    "tʃ": ["tS"], "dʒ": ["dZ"], "ts": ["ts"], "dz": ["dz"],
    "tɕ": ["ts\\"], "dʑ": ["dz\\"], "ʈʂ": ["ts`"], "ɖʐ": ["dz`"],
    "f": ["f"], "v": ["v"], "θ": ["T"], "ð": ["D"], "s": ["s"], "z": ["z"],
    "ʃ": ["S"], "ʒ": ["Z"], "ʂ": ["s`"], "ʐ": ["z`"],
    "ç": ["C\\"], "ʝ": ["j\\"], "x": ["x"], "ɣ": ["G"],
    "χ": ["X"], "ʁ": ["R\\"], "ħ": ["H\\"], "ʕ": ["9\\"],
    "h": ["h"], "ɦ": ["h\\"], "ɸ": ["p\\"], "β": ["B\\"],
    "m": ["m"], "ɱ": ["F"], "n": ["n"], "ɳ": ["n`"], "ɲ": ["J", "ny"],
    "ŋ": ["N"], "ɴ": ["N\\"], "l": ["l"], "ɭ": ["l`"], "ʎ": ["L\\"],
    "ʟ": ["K\\"], "r": ["r"], "ʀ": ["R\\"], "ɾ": ["4"],
    "ɽ": ["r`"], "ɹ": ["r\\"], "ɻ": ["r\\`"], "j": ["j"], "w": ["w"],
    "ʍ": ["W"], "ʙ": ["b\\"], "ʋ": ["V"],
    "i": ["i"], "y": ["y"], "ɨ": ["1"], "ʉ": ["}"], "ɯ": ["M"], "u": ["u"],
    "ɪ": ["I"], "ʏ": ["Y"], "ʊ": ["U"],
    "e": ["e"], "ø": ["2"], "ɘ": ["@\\"], "ɵ": ["8"], "ɤ": ["7"], "o": ["o"],
    "ə": ["@"], "ɜ": ["3"], "ɞ": ["3\\"], "ʌ": ["V"], "ɔ": ["O"],
    "æ": ["{"], "ɐ": ["6"], "a": ["a"], "ɶ": ["&"], "ɑ": ["A"], "ɒ": ["Q"],
    "ɚ": ["3`"],
    "ɑ̃": ["A~"], "ɛ̃": ["E~"], "ɔ̃": ["O~"], "œ̃": ["9~"], "ã": ["a~"],
    "ɛ": ["E"], "œ": ["9"],
    "pʰ": ["p_h"], "tʰ": ["t_h"], "kʰ": ["k_h"],
    "pʼ": ["p_>"], "tʼ": ["t_>"], "kʼ": ["k_>"],
}

# ---- 2) -X 自省探针：(声部, 含目标音的真实词, 目标 IPA)
INTROSPECTION: list[tuple[str, str, str]] = [
    ("en-us", "bit", "ɪ"), ("en-gb", "hot", "ɒ"), ("en-gb", "whale", "ʍ"),
    ("fr", "peu", "ø"), ("fr", "rue", "ʁ"), ("fr", "œuvre", "œ"),
    ("de", "hören", "ø"), ("de", "über", "y"), ("de", "Mutter", "ʊ"),
    ("cmn-latn-pinyin", "zhi", "ʈʂ"), ("cmn-latn-pinyin", "shi", "ʂ"),
    ("cmn-latn-pinyin", "ri", "ʐ"), ("cmn-latn-pinyin", "chi", "ʈʂ"),
    ("cmn-latn-pinyin", "ge", "ɤ"), ("cmn-latn-pinyin", "si", "ɿ"),
    ("hi", "ट", "ʈ"), ("hi", "ड", "ɖ"), ("hi", "ण", "ɳ"), ("hi", "ह", "ɦ"),
    ("ar", "ح", "ħ"), ("ar", "ع", "ʕ"), ("ar", "غ", "ɣ"), ("ar", "ق", "q"),
    ("ko", "으", "ɯ"), ("vi", "tư", "ɨ"), ("ru", "ты", "ɨ"),
    ("es", "elle", "ʎ"), ("it", "bagno", "ɲ"),
    ("nl", "wat", "ʋ"), ("pl", "żaba", "ʐ"),
    ("ru", "это", "ɛ"), ("ja", "う", "ɯ"),
    ("en-us", "red", "ɹ"), ("en-us", "row", "ɻ"), ("en-gb", "red", "ɹ"),
    ("en-us", "cat", "æ"), ("en-gb", "cat", "æ"),
    ("es", "pero", "ɾ"), ("es", "aya", "ʝ"),
    ("fr", "neuf", "œ"), ("fr", "un", "œ̃"),
    ("de", "fünf", "ʏ"), ("de", "müssen", "ʏ"),
    ("ta", "ட", "ʈ"), ("ta", "ண", "ɳ"),
]

# ---- 3) 控制符号（差分验证；[[...]] 内音素必须空格分隔，实测重音标记无效已剔除）
# attach=True 的控制符号（长音/声调）必须附着在前一个音素代码上（A:、A_3）
CONTROL_CANDIDATES: dict[str, list[tuple[str, str, str, bool]]] = {
    "ː": [(":", "b A: b", "b A b", True)],
    ".": [("|", "b A | t A", "b A t A", False)],
}
# 声调（Chao 调符）：附着在元音代码上，仅 tonal 声部有效
TONE_CANDIDATES: dict[str, tuple[str, str, str]] = {
    "˥": ("_1", "m A_1 m A", "m A m A"),
    "˧˥": ("_2", "m A_2 m A", "m A m A"),
    "˨˩˦": ("_3", "m A_3 m A", "m A m A"),
    "˥˩": ("_4", "m A_4 m A", "m A m A"),
    "˧": ("_5", "m A_5 m A", "m A m A"),
}


def find_espeak() -> str:
    import shutil

    for name in ("espeak-ng", "espeak-ng.exe", "espeak", "espeak.exe"):
        found = shutil.which(name)
        if found:
            return found
    for candidate in (
        r"C:\Program Files\eSpeak NG\espeak-ng.exe",
        r"C:\Program Files (x86)\eSpeak NG\espeak-ng.exe",
    ):
        if Path(candidate).exists():
            return candidate
    sys.exit("找不到 espeak-ng。请先安装：winget install eSpeak-NG.eSpeak-NG")


def ipa_of(exe: str, voice: str, phonemes: str) -> str:
    done = subprocess.run(
        [exe, "-v", voice, "--ipa", "-q", "-x", f"[[{phonemes}]]"],
        capture_output=True, text=True, timeout=TIMEOUT,
    )
    return (done.stdout or "").strip()


def x_codes(exe: str, voice: str, word: str) -> list[str]:
    done = subprocess.run(
        [exe, "-v", voice, "-X", "-q", word],
        capture_output=True, text=True, timeout=TIMEOUT,
    )
    out = done.stdout or ""
    codes: list[str] = []
    for token in re.findall(r"\[([^\[\]]+)\]", out):
        code = token.strip()
        if code and not code.startswith("_") and code not in codes:
            codes.append(code)
    return codes


def wav_size(exe: str, voice: str, phonemes: str) -> int:
    with tempfile.TemporaryDirectory() as temp:
        wav = Path(temp) / "t.wav"
        subprocess.run(
            [exe, "-v", voice, "-w", str(wav), f"[[{phonemes}]]"],
            capture_output=True, text=True, timeout=TIMEOUT,
        )
        return wav.stat().st_size if wav.exists() else -1


def base_char(symbol: str) -> str:
    return unicodedata.normalize("NFD", symbol)[0]


VOWEL_CHARS = set("aeiouyAEIOUY@&{}$20345678902\"")  # espeak 助记体系中的元音字符（近似）

def code_ok(exe: str, voice: str, symbol: str, code: str) -> bool:
    expect = base_char(symbol)
    out = ipa_of(exe, voice, code) or ipa_of(exe, voice, f"b {code} b")
    return bool(out) and expect in unicodedata.normalize("NFD", out)


def plausible_code(symbol: str, code: str) -> bool:
    """-X 自省代码过滤器：防多音素粘连（辅音目标不应含元音字符，元音目标不应含鼻/流音字母）。"""
    if len(code) > 4:
        return False
    symbol_vowel = base_char(symbol) in "aeiouyæøəɛœɐɶɑɒɜɞʌɔɪʏʊʉɨɯɤɘɚ"
    code_vowel = any(ch in VOWEL_CHARS for ch in code)
    if symbol_vowel:
        return code_vowel and not any(ch in "nmrlNMRL" for ch in code)
    return not code_vowel


def diff_ok(exe: str, voice: str, marked: str, plain: str) -> bool:
    if ipa_of(exe, voice, marked) != ipa_of(exe, voice, plain):
        return True
    a, b = wav_size(exe, voice, marked), wav_size(exe, voice, plain)
    return a > 0 and abs(a - b) > 400


def main() -> None:
    exe = find_espeak()
    version = subprocess.run([exe, "--version"], capture_output=True, text=True).stdout.strip()
    print(f"{version}\n{exe}")

    voices_out = subprocess.run([exe, "--voices"], capture_output=True, text=True).stdout or ""
    langs = set(re.findall(r"\b([a-z]{2,3}(?:-[a-z0-9A-Z]+)*)\b", voices_out))
    probe_voices = [v for v in PROBE_VOICES if v in langs]
    print(f"声部：{probe_voices}")

    # symbol -> {voice: code}（多声部并记）
    table: dict[str, dict[str, str]] = {}
    note: list[str] = []

    def record(symbol: str, voice: str, code: str) -> None:
        if code_ok(exe, voice, symbol, code):
            table.setdefault(symbol, {})[voice] = code

    # ---- 候选拼写：每个声部都试（并记）
    for symbol, codes in CANDIDATES.items():
        for voice in probe_voices:
            for code in codes:
                record(symbol, voice, code)

    # ---- -X 自省：提取代码并按所属声部验证（过滤多音素粘连）
    for voice, word, symbol in INTROSPECTION:
        if voice not in langs:
            continue
        for code in x_codes(exe, voice, word):
            if not plausible_code(symbol, code):
                continue
            before = len(table.get(symbol, {}))
            record(symbol, voice, code)
            if len(table.get(symbol, {})) > before:
                note.append(f"{symbol}: {voice} -X『{word}』-> {code}")

    covered = sorted(table)
    missing = sorted(set(CANDIDATES) - set(table))
    print(f"\n通过 {len(covered)} 个符号（多声部并记）")
    if note:
        print("自省新增：")
        for line in note:
            print(f"  {line}")
    if missing:
        print(f"仍不支持：{' '.join(missing)}")

    # ---- 控制符号与声调：逐声部差分验证，记录全部可用声部
    controls: dict[str, dict[str, str]] = {}
    attach: dict[str, bool] = {"ː": True, ".": False}
    control_voices = sorted({v for m in table.values() for v in m}) or ["en-us"]
    for symbol, tries in CONTROL_CANDIDATES.items():
        for code, marked, plain, is_attach in tries:
            for voice in control_voices:
                if diff_ok(exe, voice, marked, plain):
                    controls.setdefault(symbol, {})[voice] = code
            if symbol in controls:
                attach[symbol] = is_attach
                break
    tone_voices = [v for v in ("cmn", "cmn-latn-pinyin", "vi", "yue") if v in langs]
    for symbol, (code, marked, plain) in TONE_CANDIDATES.items():
        for voice in tone_voices:
            if diff_ok(exe, voice, marked, plain):
                controls.setdefault(symbol, {})[voice] = code
        attach[symbol] = True
    tones = sum(1 for s in TONE_CANDIDATES if s in controls)
    print(f"控制符号：{sorted(controls)}（声调 {tones}/5）")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": 3,
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "espeak": version,
        "voices": sorted({v for m in table.values() for v in m}),
        "preferred_voice": sorted(
            {v for m in table.values() for v in m}, key=lambda v: -sum(v in m for m in table.values())
        )[0] if table else "en-us",
        "map": {s: dict(sorted(vs.items())) for s, vs in sorted(table.items())},
        "controls": {s: dict(sorted(vs.items())) for s, vs in sorted(controls.items())},
        "attach": attach,
        "unsupported": missing,
    }
    OUT_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"\n已生成 {OUT_FILE}（格式 3：多声部+附着型控制）")
    if len(table) < 60:
        print("警告：通过验证的音素偏少。")


if __name__ == "__main__":
    main()
