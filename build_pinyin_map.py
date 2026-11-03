#!/usr/bin/env python3
"""生成 assets/pinyin_syllables.json：拼音音节(带声调) -> 同音汉字映射表。

一次性构建脚本，运行方式（不依赖项目 .venv）：
    uv run --no-project --with pypinyin build_pinyin_map.py

数据来源（全部本地，无网络请求）：
1. ChatTTS tokenizer 词表（模型缓存中的 tokenizer.json，BERT 中文 vocab）
   —— 候选字必须存在于词表，否则合成时会被替换成 [UNK]。
2. ChatTTS 自带的 res/homophones_map.json（项目 .venv 内）
   —— 其“值”集合是官方语料验证过 ChatTTS 能正确读音的汉字，选字时优先。
3. pypinyin（Style.TONE3 + heteronym + neutral_tone_with_five=True）
   —— 给出每个汉字的所有带调读音（如 zhong1/zhong4）。
4. pypinyin 自带词组词典 phrases_dict（4.7 万词）
   —— 统计每个读音在词语中的出现次数，用于确定“未标声调时的默认声调”
   （如 hao 默认 hao3“好”、men 默认 men5“们”，而非按词表 id 误选）。

排序规则：先选“主读音（pypinyin 首读音）与音节一致”的字，再选“官方验证读音”的字，
再按 vocab id（近似字频）从小到大。主读音优先很重要：模型看到单字会按最常见读音发音，
如果一个字把该音节只作为次读音（如「匿」的 te4），单独出现时模型会读成主读音 ni4。
每个音节保留前 3 个候选字。运行时 voice_design_server.py 取第 1 个。

默认声调（default_tone）：用户不标声调时使用。按词组词典中该读音的出现次数取最高，
无词组数据的音节回退到“首选字 vocab id 最小”的声调。

输出格式：
{
  "format": 1,
  "generated": "2026-11-02T...",
  "count": 音节数,
  "syllables": {"ni3": ["你", "尼", "泥"], ...},
  "default_tone": {"ni": "2", "hao": "3", ...}   # 未标声调时按字频取最常见声调
}
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SYL_PATTERN = re.compile(r"^[a-z]+[1-5]$")
MAX_CANDIDATES = 3


def find_tokenizer_json() -> Path | None:
    """在模型缓存里找 ChatTTS 的 tokenizer.json（跳过 .incomplete-* 目录）。"""
    pattern = "models--2Noise--ChatTTS/snapshots/*/asset/tokenizer/tokenizer.json"
    for base in (ROOT / "models" / "huggingface" / "hub",):
        candidates = sorted(base.glob(pattern))
        if candidates:
            return candidates[-1]
    return None


def find_homophones() -> Path | None:
    """找项目 .venv 里 ChatTTS 包自带的 homophones_map.json（Windows/macOS 布局）。"""
    for rel in (
        ".venv/Lib/site-packages/ChatTTS/res/homophones_map.json",
        ".venv/lib/site-packages/ChatTTS/res/homophones_map.json",
        ".venv/lib/python3.11/site-packages/ChatTTS/res/homophones_map.json",
    ):
        path = ROOT / rel
        if path.exists():
            return path
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", default=None, help="tokenizer.json 路径（默认自动搜索模型缓存）")
    parser.add_argument("--homophones", default=None, help="homophones_map.json 路径（默认自动搜索 .venv）")
    parser.add_argument("--out", default=str(ROOT / "assets" / "pinyin_syllables.json"))
    args = parser.parse_args()

    tok_path = Path(args.tokenizer) if args.tokenizer else find_tokenizer_json()
    if tok_path is None or not tok_path.exists():
        raise SystemExit(
            "找不到 tokenizer.json。请先在设计台完成一次生成（下载模型），"
            "或用 --tokenizer 手动指定路径。"
        )
    homo_path = Path(args.homophones) if args.homophones else find_homophones()
    verified: set[str] = set()
    if homo_path is not None and homo_path.exists():
        verified = set(json.loads(homo_path.read_text(encoding="utf-8")).values())
        print(f"homophones_map: {homo_path}（验证读音字 {len(verified)} 个）")
    else:
        print("未找到 homophones_map.json，将只按词表频率选字。")

    from pypinyin import Style, pinyin

    vocab: dict[str, int] = json.loads(tok_path.read_text(encoding="utf-8"))["model"]["vocab"]
    assert "儿" in vocab, "词表中没有「儿」字，无法做儿化后缀"
    char_ids: list[tuple[str, int]] = [
        (token, index)
        for token, index in vocab.items()
        if len(token) == 1 and "一" <= token <= "鿿"
    ]
    print(f"tokenizer: {tok_path}（词表 {len(vocab)}，其中汉字 {len(char_ids)} 个）")

    # syllable -> [(primary_rank, verified_rank, vocab_id, char), ...]
    table: dict[str, list[tuple[int, int, int, str]]] = {}
    unreadable = 0
    for char, vocab_id in char_ids:
        readings = pinyin(char, style=Style.TONE3, heteronym=True, neutral_tone_with_five=True)
        row = readings[0] if readings else []
        if not row:
            unreadable += 1
            continue
        primary = str(row[0])  # pypinyin 首读音 = 该字最常见的读音
        for reading in row:
            syllable = str(reading)
            if not SYL_PATTERN.match(syllable):
                continue  # 忽略非常规记号（如 ê 等），运行时输入只有 a-z
            primary_rank = 0 if syllable == primary else 1
            table.setdefault(syllable, []).append(
                (primary_rank, 0 if char in verified else 1, vocab_id, char)
            )
    if unreadable:
        print(f"pypinyin 无法注音的汉字 {unreadable} 个（已跳过）")

    syllables: dict[str, list[str]] = {}
    for syllable, entries in table.items():
        entries.sort()
        picked: list[str] = []
        seen: set[str] = set()
        for _primary_rank, _verified_rank, _vid, char in entries:
            if char in seen:
                continue
            seen.add(char)
            picked.append(char)
            if len(picked) >= MAX_CANDIDATES:
                break
        syllables[syllable] = picked

    # 默认声调：未标调时使用。优先按词组词典的读音出现次数，无数据时回退到词表 id。
    tone_marks = {"̄": "1", "́": "2", "̌": "3", "̀": "4"}

    def phrase_syllable(marked: str) -> str | None:
        """把词组词典里的带调音节（如 hǎo / me）转成 TONE3 数字形式（hao3 / me5）。"""
        letters: list[str] = []
        tone = ""
        for ch in unicodedata.normalize("NFD", marked):
            if ch in tone_marks:
                tone = tone_marks[ch]
            elif ch == "̈":
                if letters and letters[-1] == "u":
                    letters[-1] = "v"
            elif "a" <= ch <= "z":
                letters.append(ch)
            else:
                return None
        return "".join(letters) + (tone or "5")

    usage: dict[tuple[str, str], int] = {}
    from pypinyin.phrases_dict import phrases_dict

    for word, variants in phrases_dict.items():
        for variant in variants:
            for char, marked in zip(word, variant):
                syllable = phrase_syllable(str(marked))
                if syllable and SYL_PATTERN.match(syllable):
                    key = (char, syllable)
                    usage[key] = usage.get(key, 0) + 1

    # 超高频功能词（的/是/你/我……）几乎不以词组形式进入词典，词组统计会低估它们，
    # 导致不标声调时默认读音选错（如 ni 选了「尼」）。这里补一个加权读音表；
    # 每一条都在构建时用 pypinyin 反向校验，手写错误会让构建直接失败。
    common_char_readings = {
        "的": "de5", "了": "le5", "是": "shi4", "我": "wo3", "你": "ni3",
        "他": "ta1", "她": "ta1", "它": "ta1", "们": "men5", "好": "hao3",
        "不": "bu4", "这": "zhe4", "那": "na4", "就": "jiu4", "也": "ye3",
        "都": "dou1", "很": "hen3", "吗": "ma5", "吧": "ba5", "啊": "a5",
        "呢": "ne5", "和": "he2", "在": "zai4", "有": "you3", "人": "ren2",
        "什": "shen2", "一": "yi1", "二": "er4", "三": "san1", "四": "si4",
        "五": "wu3", "上": "shang4", "下": "xia4", "中": "zhong1", "大": "da4",
        "小": "xiao3", "来": "lai2", "去": "qu4", "说": "shuo1", "看": "kan4",
        "想": "xiang3", "没": "mei2", "会": "hui4", "能": "neng2", "要": "yao4",
    }
    seed_weight = 1000
    for char, syllable in common_char_readings.items():
        readings = [
            str(r)
            for r in pinyin(char, style=Style.TONE3, heteronym=True, neutral_tone_with_five=True)[0]
        ]
        if syllable not in readings:
            raise SystemExit(
                f"common_char_readings 校验失败：{char} 的读音 {readings} 不包含 {syllable}"
            )
        usage[(char, syllable)] = usage.get((char, syllable), 0) + seed_weight
    print(f"词组读音统计：{len(usage)} 个 字×读音 组合（含 {len(common_char_readings)} 个功能词加权）")

    score_by_syllable: dict[str, int] = {}
    for (char, syllable), count in usage.items():
        score_by_syllable[syllable] = score_by_syllable.get(syllable, 0) + count

    default_tone: dict[str, str] = {}
    base_best: dict[str, tuple[int, str]] = {}
    for syllable, entries in table.items():
        base = syllable[:-1]
        tone = syllable[-1]
        _pr, _vr, best_vid, _ch = entries[0]
        score = -score_by_syllable.get(syllable, 0)  # 词组次数优先（多者优先）
        fallback = best_vid  # 无词组数据时按词表 id（小者优先）
        key = (score, fallback)
        if base not in base_best or key < base_best[base][0]:
            base_best[base] = (key, tone)

    default_tone = {base: tone for base, (_key, tone) in base_best.items()}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": 1,
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "count": len(syllables),
        "syllables": dict(sorted(syllables.items())),
        "default_tone": dict(sorted(default_tone.items())),
    }
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(
        f"已生成 {out_path}：{len(syllables)} 个带调音节、"
        f"{len(default_tone)} 个基础音节。样例："
        + ", ".join(
            f"{s}={ ''.join(syllables[s]) }" for s in ("ni3", "hao3", "zhong1", "er5") if s in syllables
        )
    )


if __name__ == "__main__":
    main()
