import hashlib
import json
import time
from pathlib import Path
from typing import Optional

import logging
import requests


class SiliconFlowTTS:
    def __init__(
        self,
        api_url: str,
        api_key: str,
        model: str,
        fmt: str = "wav",  # 改为wav默认格式
        speed: float = 0.9,  # 稍微慢一点
        max_retries: int = 3,  # 增加重试次数
        timeout: int = 45,  # 增加超时时间
        *,
        gain: float = 0,  # 降低默认增益
        sample_rate: Optional[int] = None,
    ):
        self.api_url = (api_url or "").rstrip("/")
        self.api_key = api_key or ""
        self.model = model
        self.format = fmt
        self.speed = speed
        self.max_retries = max_retries
        self.timeout = timeout
        self.gain = gain
        self.sample_rate = sample_rate

    def _is_audio_response(self, r: requests.Response) -> bool:
        ct = r.headers.get("Content-Type", "").lower()
        return ct.startswith("audio/") or ct.startswith("application/octet-stream")

    def synth(self, text: str, voice: str, out_dir: Path, speed: Optional[float] = None) -> Optional[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)

        if not self.api_url or not self.api_key:
            logging.error("SiliconFlowTTS: 缺少 api_url 或 api_key")
            return None

        # 有效语速：优先使用传入值，其次使用全局默认
        eff_speed = float(speed) if speed is not None else float(self.speed)

        # 缓存 key：文本+voice+model+speed+format+gain+sample_rate
        key = hashlib.sha256(
            json.dumps(
                {
                    "t": text,
                    "v": voice,
                    "m": self.model,
                    "s": eff_speed,
                    "f": self.format,
                    "g": self.gain,
                    "sr": self.sample_rate,
                },
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()[:16]
        out_path = out_dir / f"{key}.{self.format}"
        if out_path.exists() and out_path.stat().st_size > 0:
            return out_path

        url = f"{self.api_url}/audio/speech"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "voice": voice,
            "input": text,
            "response_format": self.format,
            "speed": eff_speed,
            "gain": self.gain,
        }
        if self.sample_rate:
            payload["sample_rate"] = int(self.sample_rate)
        
        # 添加一些可能有助于完整播放的参数
        if self.format in ("mp3", "wav"):
            # 对于mp3和wav格式，确保有足够的缓冲
            payload.setdefault("quality", "high")
        
        # 确保文本以句号结尾，有助于TTS完整处理
        if text and not text.rstrip().endswith(('.', '。', '!', '！', '?', '？')):
            payload["input"] = text.rstrip() + "。"

        last_err = None
        backoff = 1.0
        for attempt in range(1, self.max_retries + 2):  # 尝试(重试N次+首次)=N+1 次
            try:
                r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.timeout)
                # 2xx
                if 200 <= r.status_code < 300:
                    if not self._is_audio_response(r):
                        # 可能是 JSON 错误
                        try:
                            err = r.json()
                        except Exception:
                            err = {"error": r.text[:200]}
                        logging.error(f"SiliconFlowTTS: 返回非音频内容，code={r.status_code}, detail={err}")
                        last_err = err
                        break
                    with open(out_path, "wb") as f:
                        f.write(r.content)
                    
                    # 验证音频文件大小，确保不是空文件或过小文件
                    if out_path.stat().st_size < 1024:  # 小于1KB可能有问题
                        logging.warning(f"SiliconFlowTTS: 生成的音频文件过小 ({out_path.stat().st_size} bytes)")
                        try:
                            out_path.unlink()
                        except Exception:
                            pass
                        last_err = "Generated audio file too small"
                        continue
                    
                    return out_path

                # 非 2xx
                err_detail = None
                try:
                    err_detail = r.json()
                except Exception:
                    err_detail = {"error": r.text[:200]}

                logging.warning(
                    f"SiliconFlowTTS: 请求失败({r.status_code}) attempt={attempt}, detail={err_detail}"
                )
                last_err = err_detail
                # 429 或 5xx 进行重试
                if r.status_code in (429,) or 500 <= r.status_code < 600:
                    if attempt <= self.max_retries:
                        time.sleep(backoff)
                        backoff = min(backoff * 2, 8)
                        continue
                break
            except Exception as e:
                logging.warning(f"SiliconFlowTTS: 网络异常 attempt={attempt}, err={e}")
                last_err = str(e)
                if attempt <= self.max_retries:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 8)
                    continue
                break

        # 失败清理
        try:
            if out_path.exists() and out_path.stat().st_size == 0:
                out_path.unlink()
        except Exception:
            pass
        logging.error(f"SiliconFlowTTS: 合成失败，已放弃。last_error={last_err}")
        return None
