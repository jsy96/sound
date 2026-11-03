#!/usr/bin/env python3
"""通用音标（espeak-ng）端到端测试：真实合成多语言音标样例。"""

from pathlib import Path

import voice_design_server as s


def check(text: str, out_name: str) -> None:
    voice, phonemes, info, errors = (None, None, None, None)
    pcm, rate = s.world_synthesize(text, 5, 0.0, 0.0)
    out = Path("outputs") / out_name
    out.parent.mkdir(exist_ok=True)
    out.write_bytes(s.wrap_wav(pcm, rate))
    seconds = len(pcm) / 2 / rate
    print(f"[OK] {text!r} -> {out.name}  {seconds:.2f}s @ {rate}Hz")
    assert seconds > 0.1, "音频过短"


def preview(text: str) -> None:
    payload = s.world_preview({"text": text})
    print(f"    预览: voice={payload['voice']} phonemes={payload['phonemes']!r} errors={payload['errors']}")


if __name__ == "__main__":
    print("== 通用音标 e2e（espeak-ng，无需加载 ChatTTS）==")
    cases = [
        ("b ɔ̃ ʒ u ʁ", "test_world_french.wav"),       # 法语味道：鼻化+小舌 r
        ("h e l o", "test_world_hello.wav"),            # 基础音
        ("m a ˥ m a ˧˥ m a ˨˩˦ m a ˥˩", "test_world_tones.wav"),  # 四声
        ("pʰ i ˧˥ tʼ o ˥˩", "test_world_asp_ej.wav"),   # 送气+喷音+声调
        ("ɑ̃ t ɛ̃ d ɔ̃", "test_world_nasals.wav"),        # 鼻化元音串
        ("b a ː t a ː", "test_world_length.wav"),        # 长音
    ]
    for text, name in cases:
        preview(text)
        check(text, name)
    print("== 声部指定（严/宽分层）==")
    for voice in ("", "fr", "cmn"):
        payload = s.world_preview({"text": "b a ʒ a", "voice": voice})
        tiers = "".join("严" if item.get("exact") else "宽" for item in payload["symbols"] if item.get("supported"))
        print(f"    voice={payload['voice'] or 'auto'} phonemes={payload['phonemes']!r} 分层={tiers} errors={payload['errors'][:1]}")
    print("== render 全链路（espeak -> ffmpeg 滤镜）==")
    data, path = s.render_audio(
        {"chinese": "b ɔ̃ ʒ u ʁ", "text_mode": "world", "base_voice": "speaker_001", "rate": 5}
    )
    print(f"[OK] render_audio -> {path.name} {len(data)} bytes")
    print("全部通过")
