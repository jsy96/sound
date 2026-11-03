#!/usr/bin/env python3
"""用真实 espeak-ng 全量验证 X-SAMPA→IPA 映射，生成 assets/espeak_ipa_map.json。

数据源：X-SAMPA 官方对照表（John C. Wells, 1995；360百科 X-SAMPA 词条收录的
小写/大写/其它符号/附加符号/图表五张表）。espeak-ng 的 [[...]] 音素输入使用
X-SAMPA 风格代码，本脚本把表中的每个代码喂给真实 espeak-ng，用 --ipa 输出比对：

  严式命中（exact）：--ipa 输出与该 IPA 符号严格一致（去掉测试框架后）；
  宽式命中（loose）：--ipa 输出仅包含该符号的基本字符（实际发音可能被
                    声部「着色」，如 cmn 下 [[r]] 实际发 ʐ——这就是宽式层）。

运行时（voice_design_server.py 通用音标模式）优先采用严式命中，选不到时才落到
宽式层并在预览中说明；未命中任何一层的符号明确报错，绝不静默近似。

用法（需已安装 espeak-ng）：
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

# X-SAMPA 小写符号表
XS_LOWER = {
    "a": "a", "b": "b", "b_<": "ɓ", "c": "c", "d": "d", "d`": "ɖ", "d_<": "ɗ",
    "e": "e", "f": "f", "g": "ɡ", "g_<": "ɠ", "h": "h", "h\\": "ɦ",
    "i": "i", "i\\": "ɨ", "j": "j", "j\\": "ʝ", "k": "k", "l": "l",
    "l`": "ɭ", "l\\": "ɺ", "m": "m", "n": "n", "n`": "ɳ", "o": "o",
    "p": "p", "p\\": "ɸ", "q": "q", "r": "r", "r`": "ɽ", "r\\": "ɹ",
    "r\\`": "ɻ", "s": "s", "s`": "ʂ", "s\\": "ɕ", "t": "t", "t`": "ʈ",
    "u": "u", "u\\": "ʉ", "v": "v", "v\\": "ʋ", "w": "w", "x": "x",
    "x\\": "ɧ", "y": "y", "z": "z", "z`": "ʐ", "z\\": "ʑ",
}
# X-SAMPA 大写符号表
XS_UPPER = {
    "A": "ɑ", "B": "β", "B\\": "ʙ", "C": "ç", "D": "ð", "E": "ɛ", "F": "ɱ",
    "G": "ɣ", "G\\": "ɢ", "G\\_<": "ʛ", "H": "ɥ", "H\\": "ʜ", "I": "ɪ",
    "I\\": "ɪ̈", "J": "ɲ", "J\\": "ɟ", "J\\_<": "ʄ", "K": "ɬ", "K\\": "ɮ",
    "L": "ʎ", "L\\": "ʟ", "M": "ɯ", "M\\": "ɰ", "N": "ŋ", "N\\": "ɴ",
    "O": "ɔ", "O\\": "ʘ", "Q": "ɒ", "R": "ʁ", "R\\": "ʀ", "S": "ʃ",
    "T": "θ", "U": "ʊ", "U\\": "ʊ̈", "V": "ʌ", "W": "ʍ", "X": "χ",
    "X\\": "ħ", "Y": "ʏ", "Z": "ʒ",
}
# X-SAMPA 其它符号表
XS_OTHER = {
    "@": "ə", "@\\": "ɘ", "{": "æ", "}": "ʉ", "1": "ɨ", "2": "ø",
    "3": "ɜ", "3\\": "ɞ", "4": "ɾ", "5": "ɫ", "6": "ɐ", "7": "ɤ",
    "8": "ɵ", "9": "œ", "&": "ɶ", "?": "ʔ", "?\\": "ʕ",
    "@`": "ɚ", "3`": "ɝ",
}
# 塞擦音与协同发音（官方图表）
XS_AFFR = {
    "ts": "ts", "dz": "dz", "tS": "tʃ", "dZ": "dʒ",
    "ts\\": "tɕ", "dz\\": "dʑ", "ts`": "ʈʂ", "dz`": "ɖʐ",
}
# 搭嘴音
XS_CLICKS = {
    "O\\": "ʘ", "|\\": "ǀ", "!\\": "ǃ", "=\\": "ǂ", "|\\|": "ǁ",
}
# 鼻化元音（~ 附加）
XS_NASAL = {
    "a~": "ã", "e~": "ẽ", "i~": "ĩ", "o~": "õ", "u~": "ũ",
    "E~": "ɛ̃", "O~": "ɔ̃", "A~": "ɑ̃", "9~": "œ̃", "2~": "ø̃", "5~": "ũ",
}
# 送气 / 喷音 / 内爆 / 腭化 / 唇化 / 咽化 / 软腭化 / 卷舌化 / 清浊 / 带儿色彩
XS_MODS = {
    "p_h": "pʰ", "t_h": "tʰ", "k_h": "kʰ", "b_h": "bʱ", "d_h": "dʱ", "g_h": "gʱ",
    "p_>": "pʼ", "t_>": "tʼ", "k_>": "kʼ", "ts_>": "tsʼ", "kS_>": "tʃʼ",
    "b_<": "ɓ", "d_<": "ɗ", "g_<": "ɠ", "G_<": "ʛ",
    "n_j": "nʲ", "d_j": "dʲ", "s_j": "sʲ", "t_j": "tʲ",
    "k_w": "kʷ", "g_w": "gʷ", "x_w": "xʷ", "s_w": "sʷ",
    "s_?": "sˤ", "d_?": "dˤ", "t_?": "tˤ",
    "l_G": "ɫ", "s_0": "s̥", "z_0": "z̥", "r_0": "r̥", "l_0": "l̥",
    "a`": "a˞", "o`": "o˞", "3`": "ɚ",
}
# 控制符号（X-SAMPA：主重音是双引号，次重音是百分号）
XS_CONTROLS = {
    "ˈ": ('"', "b A \"b A", "b A b A", False),
    "ˌ": ('%', "b A %b A", "b A b A", False),
    "ː": (":", "b A: b", "b A b", True),
    "ˑ": (":\\", "b A:\\ b", "b A b", True),
    ".": ("|", "b A | t A", "b A t A", False),
    "‿": ("-", "b A - t A", "b A t A", False),
}
XS_TONES = {
    "˥": ("_1", "m A_1 m A"), "˧˥": ("_2", "m A_2 m A"),
    "˨˩˦": ("_3", "m A_3 m A"), "˥˩": ("_4", "m A_4 m A"),
    "˧": ("_5", "m A_5 m A"), "˦˨": ("_6", "m A_6 m A"),
}

ALL_SYMBOLS: dict[str, str] = {**XS_LOWER, **XS_UPPER, **XS_OTHER, **XS_AFFR, **XS_CLICKS, **XS_NASAL, **XS_MODS}


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


# 母语词自省：非 ASCII 词语经 stdin 传入（Windows 命令行传参会损坏编码）
INTROSPECTION_NATIVE: list[tuple[str, str, str]] = [
    ("cmn", "沙", "ʂ"), ("cmn", "日", "ʐ"), ("cmn", "炸", "ʈʂ"),
    ("cmn", "是", "ʅ"), ("cmn", "自", "ɿ"), ("cmn", "茶", "ʈʂ"),
    ("cmn", "拿", "n"), ("cmn", "他", "t"), ("cmn", "大", "d"),
    ("fr", "peu", "ø"), ("fr", "rue", "ʁ"), ("fr", "neuf", "œ"),
    ("es", "elle", "ʎ"), ("it", "bagno", "ɲ"), ("es", "aya", "ʝ"),
    ("es", "pero", "ɾ"), ("en-gb", "hot", "ɒ"), ("en-us", "cat", "æ"),
]


def x_codes_stdin(exe: str, voice: str, word: str) -> list[str]:
    done = subprocess.run(
        [exe, "-v", voice, "-b", "1", "-X", "-q"],
        input=word.encode("utf-8"), capture_output=True, timeout=TIMEOUT,
    )
    out = done.stdout.decode("utf-8", errors="replace")
    codes: list[str] = []
    for token in re.findall(r"\[([^\[\]]+)\]", out):
        code = token.strip()
        if code and not code.startswith("_") and len(code) <= 4 and code not in codes:
            codes.append(code)
    return codes


def wav_bytes(exe: str, voice: str, phonemes: str) -> bytes | None:
    """合成音素串返回 WAV 字节（espeak 确定性合成：字节不同即音不同）。"""
    with tempfile.TemporaryDirectory() as temp:
        wav = Path(temp) / "t.wav"
        subprocess.run(
            [exe, "-v", voice, "-w", str(wav), f"[[{phonemes}]]"],
            capture_output=True, timeout=TIMEOUT,
        )
        if wav.exists() and wav.stat().st_size > 200:
            return wav.read_bytes()
    return None


def wav_size(exe: str, voice: str, phonemes: str) -> int:
    with tempfile.TemporaryDirectory() as temp:
        wav = Path(temp) / "t.wav"
        subprocess.run(
            [exe, "-v", voice, "-w", str(wav), f"[[{phonemes}]]"],
            capture_output=True, text=True, timeout=TIMEOUT,
        )
        return wav.stat().st_size if wav.exists() else -1


FRAME_STRIP = set("bɑ aːˈˌ ")


def norm(symbol: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFD", symbol)
        if unicodedata.combining(ch) in (0, 0x0303, 0x030C, 0x0301, 0x0300, 0x0304)
    )


def is_exact(symbol: str, out: str) -> bool:
    """严式判定：去掉框架音素（b/a 与重音号）后，输出与符号严格一致。"""
    stripped = "".join(ch for ch in norm(out) if ch not in FRAME_STRIP)
    target = norm(symbol)
    if not target:
        return False
    # 输出可能重复多遍（[[X]] 有时回声两次），取去重后的字符集合长度近似
    compact = re.sub(r"\s+", "", stripped)
    return compact == target or compact.replace(target, "", 1) == target


def base_char(symbol: str) -> str:
    return unicodedata.normalize("NFD", symbol)[0]


def main() -> None:
    exe = find_espeak()
    version = subprocess.run([exe, "--version"], capture_output=True, text=True).stdout.strip()
    print(f"{version}\n{exe}")

    voices_out = subprocess.run([exe, "--voices"], capture_output=True, text=True).stdout or ""
    langs = set(re.findall(r"\b([a-z]{2,3}(?:-[a-z0-9A-Z]+)*)\b", voices_out))
    probe_voices = [v for v in PROBE_VOICES if v in langs]
    print(f"声部：{probe_voices}")

    exact: dict[str, dict[str, str]] = {}   # 符号 -> {声部: 代码}（严式）
    loose: dict[str, dict[str, str]] = {}   # 符号 -> {声部: 代码}（宽式）
    total = len(ALL_SYMBOLS)

    for index, (code, symbol) in enumerate(sorted(ALL_SYMBOLS.items(), key=lambda kv: kv[1])):
        for voice in probe_voices:
            out_bare = ipa_of(exe, voice, code)
            out_frame = out_bare or ipa_of(exe, voice, f"a {code} a")
            if not out_frame:
                continue
            if is_exact(symbol, out_bare) or is_exact(symbol, out_frame):
                exact.setdefault(symbol, {})[voice] = code
            elif base_char(symbol) in unicodedata.normalize("NFD", out_frame):
                loose.setdefault(symbol, {})[voice] = code
        if (index + 1) % 20 == 0:
            print(f"  … {index + 1}/{total}（严式 {len(exact)} 宽式 {len(loose)}）", flush=True)

    # 母语词自省：提取代码并按所属声部做同样的分层验证。
    # -X 提供了「代码=该词中的这个音」的地面真相；当 espeak 的 --ipa 标注本身粗糙
    # （如 cmn 的 ʂ 打成字面量 s.）时，用音频字节差分裁定：代码与去标记的基础码
    # 合成字节不同 => 是独立音位，按宽式层收录并注明。
    extra_note: list[str] = []
    for voice, word, symbol in INTROSPECTION_NATIVE:
        if voice not in langs:
            continue
        for code in x_codes_stdin(exe, voice, word):
            out_bare = ipa_of(exe, voice, code)
            out_frame = out_bare or ipa_of(exe, voice, f"a {code} a")
            accepted = None
            if out_frame and (is_exact(symbol, out_bare) or is_exact(symbol, out_frame)):
                accepted = "exact"
            elif out_frame and base_char(symbol) in unicodedata.normalize("NFD", out_frame):
                accepted = "loose"
            else:
                base_code = code.rstrip(".") or code
                if base_code != code:
                    a = wav_bytes(exe, voice, f"{code} a")
                    b = wav_bytes(exe, voice, f"{base_code} a")
                    if a and b and a != b:
                        accepted = "loose(audio)"
            if accepted == "exact" and voice not in exact.get(symbol, {}):
                exact.setdefault(symbol, {})[voice] = code
                extra_note.append(f"{symbol}: {voice}『{word}』-> {code} (严式)")
            elif accepted and voice not in loose.get(symbol, {}) and voice not in exact.get(symbol, {}):
                loose.setdefault(symbol, {})[voice] = code
                extra_note.append(f"{symbol}: {voice}『{word}』-> {code} ({accepted})")
    if extra_note:
        print("自省新增：")
        for line in extra_note:
            print(f"  {line}")

    # 控制符号：全部候选声部差分验证
    controls: dict[str, dict[str, str]] = {}
    attach: dict[str, bool] = {}
    control_voices = sorted({v for m in (exact | loose).values() for v in m}) or ["en-us"]
    for symbol, tries in XS_CONTROLS.items():
        code, marked, plain, is_attach = tries
        hit = False
        for voice in control_voices[:6]:
            diff = (
                ipa_of(exe, voice, marked) != ipa_of(exe, voice, plain)
                or abs(wav_size(exe, voice, marked) - wav_size(exe, voice, plain)) > 400
            )
            if diff:
                controls.setdefault(symbol, {})[voice] = code
                hit = True
        if hit:
            attach[symbol] = is_attach
    for symbol, (code, marked) in XS_TONES.items():
        for voice in ("cmn", "cmn-latn-pinyin", "vi", "yue"):
            if voice not in langs:
                continue
            plain = marked.replace(f"_{code[1:]}", "")
            diff = (
                ipa_of(exe, voice, marked) != ipa_of(exe, voice, plain)
                or abs(wav_size(exe, voice, marked) - wav_size(exe, voice, plain)) > 400
            )
            if diff:
                controls.setdefault(symbol, {})[voice] = code
                attach[symbol] = True

    covered = sorted(set(exact) | set(loose))
    missing = sorted(set(ALL_SYMBOLS.values()) - set(covered))
    tones_ok = sum(1 for s in XS_TONES if s in controls)
    print(f"\n严式命中 {len(exact)} 个符号；宽式新增 {len(set(loose) - set(exact))} 个；共 {len(covered)}")
    print(f"控制符号 {sorted(controls)}（声调 {tones_ok}/{len(XS_TONES)}）")
    if missing:
        print(f"未命中（报错处理）：{' '.join(missing)}")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": 4,
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "espeak": version,
        "voices": sorted({v for m in (exact | loose).values() for v in m}),
        "exact": {s: dict(sorted(vs.items())) for s, vs in sorted(exact.items())},
        "loose": {s: dict(sorted(vs.items())) for s, vs in sorted(loose.items())},
        "controls": {s: dict(sorted(vs.items())) for s, vs in sorted(controls.items())},
        "attach": attach,
        "unsupported": missing,
    }
    OUT_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"已生成 {OUT_FILE}（格式 4：X-SAMPA 全量 + 严式/宽式分层）")


if __name__ == "__main__":
    main()
