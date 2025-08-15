# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import logging
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api.message_components import Record, Plain
from astrbot.core.config.astrbot_config import AstrBotConfig
from astrbot.api.provider import LLMResponse

from .emotion.infer import EMOTIONS
from .emotion.classifier import HeuristicClassifier  # LLMClassifier 不再使用
from .tts.provider_siliconflow import SiliconFlowTTS
from .utils.audio import ensure_dir, cleanup_dir

CONFIG_FILE = Path(__file__).parent / "config.json"  # 旧版本地文件，作为迁移来源
TEMP_DIR = Path(__file__).parent / "temp"


@dataclass
class SessionState:
    last_ts: float = 0.0
    pending_emotion: Optional[str] = None  # 基于隐藏标记的待用情绪


@register("astrabot_plugin_tts_emotion_router", "木有知", "按情绪路由到不同音色的TTS插件", "0.1.0")
class TTSEmotionRouter(Star):
    def __init__(self, context: Context, config: Optional[dict] = None):
        super().__init__(context)
        # 1) 首选面板生成的插件配置（data/config/tts_emotion_router_config.json）
        #    当 _conf_schema.json 存在时，StarManager 会传入 AstrBotConfig
        if isinstance(config, AstrBotConfig):
            self.config = config
            # 若是首次创建且旧版本地 config.json 存在，则迁移一次
            try:
                if getattr(self.config, "first_deploy", False) and CONFIG_FILE.exists():
                    disk = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
                    # 仅拷贝已知字段，避免脏键
                    for k in [
                        "global_enable","enabled_sessions","disabled_sessions","prob","text_limit","cooldown","allow_mixed","api","voice_map","emotion","speed_map"
                    ]:
                        if k in disk:
                            self.config[k] = disk[k]
                    self.config.save_config()
            except Exception:
                pass
        else:
            # 兼容旧版：直接读写插件目录下的 config.json
            self.config = self._load_config(config or {})

        api = self.config.get("api", {})
        self.tts = SiliconFlowTTS(api.get("url", ""), api.get("key", ""), api.get("model", "gpt-tts-pro"), api.get("format", "wav"), float(api.get("speed", 1.0)))
        self.voice_map: Dict[str, str] = self.config.get("voice_map", {})
        self.speed_map: Dict[str, float] = self.config.get("speed_map", {}) or {}
        self.global_enable: bool = bool(self.config.get("global_enable", True))
        self.enabled_sessions: List[str] = list(self.config.get("enabled_sessions", []))
        self.disabled_sessions: List[str] = list(self.config.get("disabled_sessions", []))
        self.prob: float = float(self.config.get("prob", 0.35))
        self.text_limit: int = int(self.config.get("text_limit", 80))
        self.cooldown: int = int(self.config.get("cooldown", 20))
        self.allow_mixed: bool = bool(self.config.get("allow_mixed", False))
        # 情绪分类：仅启发式 + 隐藏标记
        emo_cfg = self.config.get("emotion", {}) or {}
        self.heuristic_cls = HeuristicClassifier()
        # 标记驱动配置（不与表情包插件冲突：仅识别 [EMO:happy] 这类专属标记）
        marker_cfg = (emo_cfg.get("marker") or {}) if isinstance(emo_cfg, dict) else {}
        self.emo_marker_enable: bool = bool(marker_cfg.get("enable", False))
        self.emo_marker_tag: str = str(marker_cfg.get("tag", "EMO"))
        try:
            tag = re.escape(self.emo_marker_tag)
            self._emo_marker_re = re.compile(rf"\[\s*{tag}\s*:\s*(happy|sad|angry|neutral)\s*\]", re.I)
        except Exception:
            self._emo_marker_re = None

        self._session_state: Dict[str, SessionState] = {}
        ensure_dir(TEMP_DIR)
        cleanup_dir(TEMP_DIR, ttl_seconds=6*3600)

    # ---------------- Config helpers -----------------
    def _load_config(self, cfg: dict) -> dict:
        # 合并磁盘config与传入config，便于热更
        try:
            if CONFIG_FILE.exists():
                disk = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            else:
                disk = {}
        except Exception:
            disk = {}
        merged = {**disk, **(cfg or {})}
        try:
            CONFIG_FILE.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        return merged

    def _save_config(self):
        # 面板配置优先保存到 data/config/tts_emotion_router_config.json
        if isinstance(self.config, AstrBotConfig):
            self.config.save_config()
        else:
            try:
                CONFIG_FILE.write_text(json.dumps(self.config, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass

    def _sess_id(self, event: AstrMessageEvent) -> str:
        gid = ""
        try:
            gid = event.get_group_id()
        except Exception:
            gid = ""
        if gid:
            return f"group_{gid}"
        return f"user_{event.get_sender_id()}"

    def _is_session_enabled(self, sid: str) -> bool:
        if self.global_enable:
            return sid not in self.disabled_sessions
        return sid in self.enabled_sessions

    # ---------------- LLM 标记解析（避免标签外显） -----------------
    @filter.on_llm_response(priority=100)
    async def on_llm_response(self, event: AstrMessageEvent, response: LLMResponse):
        if not self.emo_marker_enable:
            return
        text = getattr(response, "completion_text", None)
        if not (text and self._emo_marker_re):
            return
        # 解析并移除自有标记，如 [EMO:happy]
        m = self._emo_marker_re.search(text)
        if not m:
            return
        label = (m.group(1) or "").lower()
        if label in EMOTIONS:
            sid = self._sess_id(event)
            st = self._session_state.setdefault(sid, SessionState())
            st.pending_emotion = label
        # 把自有标记移除，避免用户看到；不处理其它插件的标记
        new_text = self._emo_marker_re.sub("", text)
        response.completion_text = new_text.strip()

    # ---------------- Commands -----------------
    @filter.command("tts_global_on", priority=1)
    async def tts_global_on(self, event: AstrMessageEvent):
        self.global_enable = True
        self.config["global_enable"] = True
        self._save_config()
        yield event.plain_result("TTS 全局：开启（黑名单模式）")

    @filter.command("tts_global_off", priority=1)
    async def tts_global_off(self, event: AstrMessageEvent):
        self.global_enable = False
        self.config["global_enable"] = False
        self._save_config()
        yield event.plain_result("TTS 全局：关闭（白名单模式）")

    @filter.command("tts_on", priority=1)
    async def tts_on(self, event: AstrMessageEvent):
        sid = self._sess_id(event)
        if self.global_enable:
            if sid in self.disabled_sessions:
                self.disabled_sessions.remove(sid)
        else:
            if sid not in self.enabled_sessions:
                self.enabled_sessions.append(sid)
        self.config["enabled_sessions"] = self.enabled_sessions
        self.config["disabled_sessions"] = self.disabled_sessions
        self._save_config()
        yield event.plain_result("本会话TTS：开启")

    @filter.command("tts_off", priority=1)
    async def tts_off(self, event: AstrMessageEvent):
        sid = self._sess_id(event)
        if self.global_enable:
            if sid not in self.disabled_sessions:
                self.disabled_sessions.append(sid)
        else:
            if sid in self.enabled_sessions:
                self.enabled_sessions.remove(sid)
        self.config["enabled_sessions"] = self.enabled_sessions
        self.config["disabled_sessions"] = self.disabled_sessions
        self._save_config()
        yield event.plain_result("本会话TTS：关闭")

    @filter.command("tts_prob", priority=1)
    async def tts_prob(self, event: AstrMessageEvent, *, value: Optional[str] = None):
        try:
            if value is None:
                raise ValueError
            v = float(value)
            assert 0.0 <= v <= 1.0
            self.prob = v
            self.config["prob"] = v
            self._save_config()
            yield event.plain_result(f"TTS概率已设为 {v}")
        except Exception:
            yield event.plain_result("用法：tts_prob 0~1，如 0.35")

    @filter.command("tts_limit", priority=1)
    async def tts_limit(self, event: AstrMessageEvent, *, value: Optional[str] = None):
        try:
            if value is None:
                raise ValueError
            v = int(value)
            assert v >= 0
            self.text_limit = v
            self.config["text_limit"] = v
            self._save_config()
            yield event.plain_result(f"TTS字数上限已设为 {v}")
        except Exception:
            yield event.plain_result("用法：tts_limit <非负整数>")

    @filter.command("tts_cooldown", priority=1)
    async def tts_cooldown(self, event: AstrMessageEvent, *, value: Optional[str] = None):
        try:
            if value is None:
                raise ValueError
            v = int(value)
            assert v >= 0
            self.cooldown = v
            self.config["cooldown"] = v
            self._save_config()
            yield event.plain_result(f"TTS冷却时间已设为 {v}s")
        except Exception:
            yield event.plain_result("用法：tts_cooldown <非负整数(秒)>")

    @filter.command("tts_status", priority=1)
    async def tts_status(self, event: AstrMessageEvent):
        sid = self._sess_id(event)
        mode = "黑名单(默认开)" if self.global_enable else "白名单(默认关)"
        enabled = self._is_session_enabled(sid)
        yield event.plain_result(f"模式: {mode}\n当前会话: {'启用' if enabled else '禁用'}\nprob={self.prob}, limit={self.text_limit}, cooldown={self.cooldown}s")

    # ---------------- Core hook -----------------
    @filter.on_decorating_result()
    async def on_decorating_result(self, event: AstrMessageEvent):
        sid = self._sess_id(event)
        if not self._is_session_enabled(sid):
            # 即使本会话未启用，也不影响上游 on_llm_response 对标签的移除
            return

        # 冷却
        st = self._session_state.setdefault(sid, SessionState())
        now = time.time()
        if self.cooldown > 0 and (now - st.last_ts) < self.cooldown:
            return

        # 结果链
        result = event.get_result()
        if not result or not result.chain:
            return

        # 是否允许混合
        if not self.allow_mixed and any(not isinstance(c, Plain) for c in result.chain):
            return

        # 拼接纯文本
        text_parts = [c.text.strip() for c in result.chain if isinstance(c, Plain) and c.text.strip()]
        if not text_parts:
            return
        text = " ".join(text_parts)

        # 先保障隐藏标记不外显（仅处理本插件自有标记）
        if self._emo_marker_re:
            text = self._emo_marker_re.sub("", text).strip()

        # 过滤链接/文件等提示性内容，避免朗读
        if re.search(r"(https?://|www\.|\[图片\]|\[文件\]|\[转发\]|\[引用\])", text, re.I):
            return

        # 长度限制
        if self.text_limit > 0 and len(text) > self.text_limit:
            return

        # 随机概率
        if random.random() > self.prob:
            return

        # 情绪选择：优先使用隐藏标记 -> 启发式
        if st.pending_emotion in EMOTIONS:
            emotion = st.pending_emotion
            st.pending_emotion = None
        else:
            emotion = self.heuristic_cls.classify(text, context=None)

        voice = self.voice_map.get(emotion) or self.voice_map.get("neutral")
        if not voice:
            logging.warning("No voice mapped for emotion=%s", emotion)
            return

        # 依据情绪选择语速（未配置则为 None -> 使用默认）
        speed_override = None
        try:
            if isinstance(self.speed_map, dict):
                v = self.speed_map.get(emotion)
                if v is None:
                    v = self.speed_map.get("neutral")
                if v is not None:
                    speed_override = float(v)
        except Exception:
            speed_override = None

        out_dir = TEMP_DIR / sid
        ensure_dir(out_dir)
        audio_path = self.tts.synth(text, voice, out_dir, speed=speed_override)
        if not audio_path:
            logging.error("TTS调用失败，降级为文本")
            return

        st.last_ts = time.time()
        result.chain = [Record(file=str(audio_path))]
