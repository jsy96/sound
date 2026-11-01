#!/usr/bin/env python3
"""本地中文多维声音设计台后端。"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import threading
import traceback
import uuid
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parent
UI_FILE = ROOT / "voice_design_ui.html"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models" / "huggingface"
SPEAKER_FILE = ROOT / "data" / "chattts_speakers.json"
VOICE_COUNT = 60
VOICE_CARDS = [
    {"id": f"speaker_{index:03d}", "name": f"普通话音色 {index:02d}"}
    for index in range(1, VOICE_COUNT + 1)
]


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

    def synthesize(self, text: str, voice_id: str, rate: int):
        self._load()
        if voice_id not in self._speakers:
            raise RuntimeError("请选择列表中的普通话音色。")

        # ChatTTS has shared model state. Serialize inference to avoid concurrent
        # requests corrupting its random-generator or decoder state.
        with self._infer_lock:
            assert self._chat is not None and self._chattts is not None
            assert self._numpy is not None
            speed = int(clamp(rate, 1, 9, 5))
            seed_text = f"{voice_id}\0{text}\0{rate}".encode("utf-8")
            seed = int(hashlib.sha256(seed_text).hexdigest()[:8], 16)
            infer_params = self._chattts.Chat.InferCodeParams(
                spk_emb=self._speakers[voice_id],
                prompt=f"[speed_{speed}]",
                manual_seed=seed,
                temperature=0.3,
                top_P=0.7,
                top_K=20,
                show_tqdm=False,
            )
            audio_list = self._chat.infer(
                text,
                params_infer_code=infer_params,
                lang="zh",
                do_text_normalization=True,
                do_homophone_replacement=True,
            )
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
        raise RuntimeError("请输入要朗读的中文文本。")
    voice_id = str(params.get("base_voice", "speaker_001"))
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
        pcm, sample_rate = ENGINE.synthesize(text, voice_id, rate)
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


class Handler(BaseHTTPRequestHandler):
    server_version = "VoiceDesign/2.0"

    def _send_json(self, status: int, payload: dict[str, object]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

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
        if parsed.path == "/api/voices":
            self._send_json(200, {"voices": VOICE_CARDS, "engine": ENGINE.status()})
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
        if self.path != "/api/render":
            self.send_error(404)
            return
        try:
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
        except Exception as exc:  # pragma: no cover
            print(f"Render request failed: {exc}", flush=True)
            traceback.print_exc()
            self._send_json(500, {"error": str(exc) or "声音生成失败，请查看服务窗口日志。"})

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
