#!/usr/bin/env python3
"""拼音模式端到端测试：真实加载 ChatTTS 并合成，验证 WAV 时长合理。"""

import wave
from pathlib import Path

import voice_design_server as s


def check(text: str, mode: str, out_name: str) -> None:
    pcm, rate = s.ENGINE.synthesize(text, 5, {"kind": "fixed", "id": "speaker_001"}, mode=mode)
    out = Path("outputs") / out_name
    out.parent.mkdir(exist_ok=True)
    with wave.open(str(out), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(pcm)
    seconds = len(pcm) / 2 / rate
    print(f"[OK] {mode} {text!r} -> {out.name}  {seconds:.2f}s @ {rate}Hz  {len(pcm)} bytes")
    assert seconds > 0.3, "音频过短"


if __name__ == "__main__":
    print("加载模型（首次约 1-2 分钟）…")
    # 1) 拼音模式：正常音节
    check("ni3 hao3", "pinyin", "test_pinyin_nihao.wav")
    # 2) 拼音模式：生造音节组合（不构成任何真实词语）
    check("za2 ka1 te4 ke1 miu4 fo2", "pinyin", "test_pinyin_nonsense.wav")
    # 3) 拼音模式：儿化 + 标点
    check("huar1 bi3 jiao4 hao3，dui4 ba5。", "pinyin", "test_pinyin_erhua.wav")
    # 4) 回归：中文模式（走 refine_text 原路径）
    check("你好，这是一个回归测试。", "zh", "test_pinyin_regression_zh.wav")
    # 5) 错误路径：无效声调应报错且不加载推理
    try:
        s.ENGINE.synthesize("shi6", 5, {"kind": "fixed", "id": "speaker_001"}, mode="pinyin")
        raise SystemExit("[FAIL] shi6 应当报错")
    except RuntimeError as exc:
        print(f"[OK] 错误提示：{exc}")
    print("全部通过")
