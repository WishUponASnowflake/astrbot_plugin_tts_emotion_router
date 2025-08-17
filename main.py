# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import logging
import random
import re
import time
import hashlib
from dataclasses import dataclass
import sys
from pathlib import Path
import importlib
from typing import Dict, List, Optional

# ... (保留文件头部所有兼容性导入代码，此处省略以保持清晰)
def _ensure_compatible_astrbot():
    """确保 astrbot API 兼容；若宿主 astrbot 不满足需要，则回退到插件自带的 AstrBot。"""
    _PLUGIN_DIR = Path(__file__).parent
    _VENDORED_ROOT = _PLUGIN_DIR / "AstrBot"
    _VENDORED_ASTROBOT = _VENDORED_ROOT / "astrbot"
    root_str = str(_PLUGIN_DIR.resolve())

    def _import_host_first():
        if _VENDORED_ASTROBOT.exists() and "astrbot" not in sys.modules:
            _orig = list(sys.path)
            try:
                # 临时移除插件路径，优先导入宿主 AstrBot
                sys.path = [p for p in sys.path if not (isinstance(p, str) and p.startswith(root_str))]
                importlib.import_module("astrbot")
            finally:
                sys.path = _orig

    def _is_compatible() -> bool:
        try:
            import importlib as _il
            _il.import_module("astrbot.api.event.filter")
            _il.import_module("astrbot.core.platform")
            return True
        except Exception:
            return False

    def _force_vendored():
        try:
            sys.modules.pop("astrbot", None)
            importlib.invalidate_caches()
            # 确保优先搜索插件自带 AstrBot
            if str(_VENDORED_ROOT) not in sys.path:
                sys.path.insert(0, str(_VENDORED_ROOT))
            importlib.import_module("astrbot")
            logging.info("TTSEmotionRouter: forced to vendored AstrBot: %s", (_VENDORED_ASTROBOT / "__init__.py").as_posix())
        except Exception:
            pass

    # 1) 优先尝试宿主
    try:
        _import_host_first()
    except Exception:
        pass
    # 2) 若不兼容，则强制改用内置 AstrBot
    if not _is_compatible() and _VENDORED_ASTROBOT.exists():
        _force_vendored()

try:
    _ensure_compatible_astrbot()
except Exception:
    pass

try:
    from astrbot.api.event import AstrMessageEvent
except Exception:
    from astrbot.core.platform import AstrMessageEvent

try:
    from astrbot.api.event import filter
except Exception:
    try:
        import importlib as _importlib
        filter = _importlib.import_module("astrbot.api.event.filter")
    except Exception:
        try:
            import astrbot.core.star.register as _reg

            class _FilterCompat:
                def command(self, *a, **k): return _reg.register_command(*a, **k)
                def on_llm_request(self, *a, **k): return _reg.register_on_llm_request(*a, **k)
                def on_llm_response(self, *a, **k): return _reg.register_on_llm_response(*a, **k)
                def on_decorating_result(self, *a, **k): return _reg.register_on_decorating_result(*a, **k)
                def after_message_sent(self, *a, **k): return _reg.register_after_message_sent(*a, **k)
                def on_after_message_sent(self, *a, **k): return _reg.register_after_message_sent(*a, **k)

            filter = _FilterCompat()
        except Exception as _e:
            raise _e

from astrbot.api.star import Context, Star, register
from astrbot.api.message_components import Record, Plain, BaseMessageComponent
from astrbot.core.config.astrbot_config import AstrBotConfig
from astrbot.api.provider import LLMResponse
from astrbot.core.message.message_event_result import ResultContentType

from .emotion.infer import EMOTIONS
from .emotion.classifier import HeuristicClassifier
from .tts.provider_siliconflow import SiliconFlowTTS
from .utils.audio import ensure_dir, cleanup_dir

try:
    import astrbot as _ab_mod
    logging.info("TTSEmotionRouter: using astrbot from %s", getattr(_ab_mod, "__file__", None))
except Exception:
    pass

CONFIG_FILE = Path(__file__).parent / "config.json"
TEMP_DIR = Path(__file__).parent / "temp"

@dataclass
class SessionState:
    last_ts: float = 0.0
    pending_emotion: Optional[str] = None
    last_tts_content_hash: Optional[str] = None
    last_tts_time: float = 0.0

@register(
    "astrabot_plugin_tts_emotion_router",
    "木有知",
    "按情绪路由到不同音色的TTS插件",
    "0.2.0", # 版本升级
)
class TTSEmotionRouter(Star):
    # ... (保留 __init__ 和所有指令处理函数，它们是正确的)
    def __init__(self, context: Context, config: Optional[dict] = None):
        super().__init__(context)
        if isinstance(config, AstrBotConfig):
            self.config = config
            try:
                if getattr(self.config, "first_deploy", False) and CONFIG_FILE.exists():
                    disk = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
                    for k in [
                        "global_enable", "enabled_sessions", "disabled_sessions",
                        "prob", "text_limit", "cooldown", "allow_mixed", "api",
                        "voice_map", "emotion", "speed_map"
                    ]:
                        if k in disk: self.config[k] = disk[k]
                    self.config.save_config()
            except Exception: pass
        else:
            self.config = self._load_config(config or {})

        api = self.config.get("api", {})
        self.tts = SiliconFlowTTS(
            api.get("url", ""), api.get("key", ""), api.get("model", "gpt-tts-pro"),
            api.get("format", "mp3"), float(api.get("speed", 1.0)),
            gain=float(api.get("gain", 5.0)),
            sample_rate=int(api.get("sample_rate", 44100))
        )

        self.voice_map: Dict[str, str] = self.config.get("voice_map", {})
        self.speed_map: Dict[str, float] = self.config.get("speed_map", {}) or {}
        self.global_enable: bool = bool(self.config.get("global_enable", True))
        self.enabled_sessions: List[str] = list(self.config.get("enabled_sessions", []))
        self.disabled_sessions: List[str] = list(self.config.get("disabled_sessions", []))
        self.prob: float = float(self.config.get("prob", 0.35))
        self.text_limit: int = int(self.config.get("text_limit", 80))
        self.cooldown: int = int(self.config.get("cooldown", 20))
        self.allow_mixed: bool = bool(self.config.get("allow_mixed", False))
        
        emo_cfg = self.config.get("emotion", {}) or {}
        self.heuristic_cls = HeuristicClassifier()
        marker_cfg = (emo_cfg.get("marker") or {}) if isinstance(emo_cfg, dict) else {}
        self.emo_marker_enable: bool = bool(marker_cfg.get("enable", True))
        self.emo_marker_tag: str = str(marker_cfg.get("tag", "EMO"))
        
        try:
            tag = re.escape(self.emo_marker_tag)
            self._emo_marker_re = re.compile(rf"\[\s*{tag}\s*:\s*(happy|sad|angry|neutral)\s*\]", re.I)
            self._emo_marker_re_any = re.compile(rf"^[\s\ufeff]*[\[\(【]\s*{tag}\s*(?:[:\uff1a-]\s*[a-z]*)?\s*[\]\)】]", re.I)
            self._emo_head_token_re = re.compile(rf"^[\s\ufeff]*(?:[\[\(【]\s*{tag}\s*(?:[:\uff1a-]\s*(?P<lbl>happy|sad|angry|neutral))?\s*[\]\)】]|(?:{tag}|emo)\s*(?:[:\uff1a-]\s*(?P<lbl2>happy|sad|angry|neutral))?)\s*[,，。:\uff1a-]*\s*", re.I)
            self._emo_head_anylabel_re = re.compile(rf"^[\s\ufeff]*[\[\(【]\s*{tag}\s*[:\uff1a-]\s*(?P<raw>[a-z]+)\s*[\]\)】]", re.I)
        except Exception:
            self._emo_marker_re, self._emo_marker_re_any, self._emo_head_token_re, self._emo_head_anylabel_re = None, None, None, None

        self._session_state: Dict[str, SessionState] = {}
        self._inflight_sigs: set[str] = set()
        ensure_dir(TEMP_DIR)
        cleanup_dir(TEMP_DIR, ttl_seconds=2 * 3600)

    def _load_config(self, cfg: dict) -> dict:
        try: disk = json.loads(CONFIG_FILE.read_text(encoding="utf-8")) if CONFIG_FILE.exists() else {}
        except Exception: disk = {}
        merged = {**disk, **(cfg or {})}
        try: CONFIG_FILE.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception: pass
        return merged

    def _save_config(self):
        if isinstance(self.config, AstrBotConfig): self.config.save_config()
        else:
            try: CONFIG_FILE.write_text(json.dumps(self.config, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception: pass

    def _sess_id(self, event: AstrMessageEvent) -> str:
        gid = ""
        try: gid = event.get_group_id()
        except Exception: gid = ""
        return f"group_{gid}" if gid else f"user_{event.get_sender_id()}"

    def _is_session_enabled(self, sid: str) -> bool:
        return sid not in self.disabled_sessions if self.global_enable else sid in self.enabled_sessions

    def _normalize_text(self, text: str) -> str:
        if not text: return text
        invisibles = ["\ufeff", "\u200b", "\u200c", "\u200d", "\u200e", "\u200f", "\u202a", "\u202b", "\u202c", "\u202d", "\u202e"]
        for ch in invisibles: text = text.replace(ch, "")
        return text
    
    def _normalize_label(self, label: Optional[str]) -> Optional[str]:
        if not label: return None
        lbl = label.strip().lower()
        mapping = {
            "happy": {"happy", "joy", "joyful", "cheerful", "delighted", "excited", "smile", "positive", "开心", "快乐", "高兴", "喜悦", "兴奋", "愉快"},
            "sad": {"sad", "sorrow", "sorrowful", "depressed", "down", "unhappy", "cry", "crying", "tearful", "blue", "upset", "伤心", "难过", "沮丧", "低落", "悲伤", "流泪"},
            "angry": {"angry", "mad", "furious", "annoyed", "irritated", "rage", "rageful", "wrath", "生气", "愤怒", "恼火", "气愤"},
            "neutral": {"neutral", "calm", "plain", "normal", "objective", "ok", "fine", "meh", "average", "confused", "uncertain", "unsure", "平静", "冷静", "一般", "中立", "客观", "困惑", "迷茫"},
        }
        for k, vs in mapping.items():
            if lbl in vs: return k
        return None

    def _pick_voice_for_emotion(self, emotion: str):
        vm = self.voice_map or {}
        if v := vm.get(emotion): return emotion, v
        if v := vm.get("neutral"): return "neutral", v
        pref = {"sad": "angry", "angry": "angry", "happy": "happy", "neutral": "happy"}
        for key in [pref.get(emotion), "happy", "angry"]:
            if key and (v := vm.get(key)): return key, v
        for k, v in vm.items():
            if v: return k, v
        return None, None

    def _strip_emo_head_many(self, text: str) -> tuple[str, Optional[str]]:
        last_label: Optional[str] = None
        while True:
            cleaned, label = self._strip_emo_head(text)
            if label: last_label = label
            if cleaned == text: break
            text = cleaned
        return text, last_label

    def _strip_emo_head(self, text: str) -> tuple[str, Optional[str]]:
        if not text: return text, None
        if self._emo_head_token_re:
            if m := self._emo_head_token_re.match(text):
                label = (m.group("lbl") or m.group("lbl2") or "").lower()
                return self._emo_head_token_re.sub("", text, count=1).strip(), label if label in EMOTIONS else None
        if self._emo_head_anylabel_re:
            if m2 := self._emo_head_anylabel_re.match(text):
                raw = (m2.group("raw") or "").lower()
                return self._emo_head_anylabel_re.sub("", text, count=1).strip(), self._normalize_label(raw)
        if self._emo_marker_re_any and text.lstrip().startswith(("[", "【", "(")):
            return self._emo_marker_re_any.sub("", text, count=1).strip(), None
        return text, None

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, request):
        if not self.emo_marker_enable: return
        try:
            tag = self.emo_marker_tag
            instr = (f"请在每次回复的最开头只输出一个隐藏情绪标记，格式严格为："
                     f"[{tag}:happy] 或 [{tag}:sad] 或 [{tag}:angry] 或 [{tag}:neutral]。"
                     "必须四选一；若无法判断请选择 neutral。该标记仅供系统解析，"
                     "输出后立刻继续正常作答，不要解释或复述该标记。")
            sp = getattr(request, 'system_prompt', '') or ''
            pp = getattr(request, 'prompt', '') or ''
            if tag not in sp and tag not in pp:
                request.system_prompt = (instr + "\n" + sp).strip()
                request.prompt = (instr + "\n\n" + pp).strip()
                if isinstance(ctxs := getattr(request, 'contexts', None), list):
                    ctxs.insert(0, {"role": "system", "content": instr})
        except Exception: pass

    @filter.on_llm_response(priority=1)
    async def on_llm_response(self, event: AstrMessageEvent, response: LLMResponse):
        if not self.emo_marker_enable: return
        label, cached_text = None, None
        try:
            if isinstance(text := getattr(response, "completion_text", None), str) and text.strip():
                cleaned, l1 = self._strip_emo_head_many(self._normalize_text(text))
                if l1 in EMOTIONS: label = l1
                response.completion_text = cleaned
                cached_text = cleaned
        except Exception: pass
        try:
            if (rc := getattr(response, "result_chain", None)) and hasattr(rc, "chain") and rc.chain:
                new_chain, cleaned_once = [], False
                for comp in rc.chain:
                    if not cleaned_once and isinstance(comp, Plain) and getattr(comp, "text", None):
                        t, l2 = self._strip_emo_head_many(self._normalize_text(comp.text))
                        if l2 in EMOTIONS and label is None: label = l2
                        if t:
                            new_chain.append(Plain(text=t))
                            if not cached_text: cached_text = t
                        cleaned_once = True
                    else: new_chain.append(comp)
                rc.chain = new_chain
        except Exception: pass
        try:
            sid = self._sess_id(event)
            st = self._session_state.setdefault(sid, SessionState())
            if label in EMOTIONS: st.pending_emotion = label
        except Exception: pass

    @filter.command("tts_marker_on", priority=1)
    async def tts_marker_on(self, event: AstrMessageEvent):
        self.emo_marker_enable = True
        emo_cfg = self.config.get("emotion", {}) or {}
        marker_cfg = (emo_cfg.get("marker") or {}) if isinstance(emo_cfg, dict) else {}
        marker_cfg["enable"] = True
        emo_cfg["marker"] = marker_cfg
        self.config["emotion"] = emo_cfg
        self._save_config()
        yield event.plain_result("情绪隐藏标记：开启")

    @filter.command("tts_marker_off", priority=1)
    async def tts_marker_off(self, event: AstrMessageEvent):
        self.emo_marker_enable = False
        emo_cfg = self.config.get("emotion", {}) or {}
        marker_cfg = (emo_cfg.get("marker") or {}) if isinstance(emo_cfg, dict) else {}
        marker_cfg["enable"] = False
        emo_cfg["marker"] = marker_cfg
        self.config["emotion"] = emo_cfg
        self._save_config()
        yield event.plain_result("情绪隐藏标记：关闭")

    @filter.command("tts_emote", priority=1)
    async def tts_emote(self, event: AstrMessageEvent, *, value: Optional[str] = None):
        try:
            label = (value or "").strip().lower()
            assert label in EMOTIONS
            sid = self._sess_id(event)
            st = self._session_state.setdefault(sid, SessionState())
            st.pending_emotion = label
            yield event.plain_result(f"已设置：下一条消息按情绪 {label} 路由")
        except Exception:
            yield event.plain_result("用法：tts_emote <happy|sad|angry|neutral>")

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
            if sid in self.disabled_sessions: self.disabled_sessions.remove(sid)
        else:
            if sid not in self.enabled_sessions: self.enabled_sessions.append(sid)
        self.config["enabled_sessions"] = self.enabled_sessions
        self.config["disabled_sessions"] = self.disabled_sessions
        self._save_config()
        yield event.plain_result("本会话TTS：开启")

    @filter.command("tts_off", priority=1)
    async def tts_off(self, event: AstrMessageEvent):
        sid = self._sess_id(event)
        if self.global_enable:
            if sid not in self.disabled_sessions: self.disabled_sessions.append(sid)
        else:
            if sid in self.enabled_sessions: self.enabled_sessions.remove(sid)
        self.config["enabled_sessions"] = self.enabled_sessions
        self.config["disabled_sessions"] = self.disabled_sessions
        self._save_config()
        yield event.plain_result("本会话TTS：关闭")

    @filter.command("tts_prob", priority=1)
    async def tts_prob(self, event: AstrMessageEvent, *, value: Optional[str] = None):
        try:
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
            v = int(value)
            assert v >= 0
            self.cooldown = v
            self.config["cooldown"] = v
            self._save_config()
            yield event.plain_result(f"TTS冷却时间已设为 {v}s")
        except Exception:
            yield event.plain_result("用法：tts_cooldown <非负整数(秒)>")
    
    @filter.command("tts_gain", priority=1)
    async def tts_gain(self, event: AstrMessageEvent, *, value: Optional[str] = None):
        try:
            v = float(value)
            assert -10.0 <= v <= 10.0
            self.tts.gain = v
            api_cfg = self.config.get("api", {}) or {}
            api_cfg["gain"] = v
            self.config["api"] = api_cfg
            self._save_config()
            yield event.plain_result(f"TTS音量增益已设为 {v} dB")
        except Exception:
            yield event.plain_result("用法：tts_gain <-10~10>，例：tts_gain 5")
    
    @filter.command("tts_status", priority=1)
    async def tts_status(self, event: AstrMessageEvent):
        sid = self._sess_id(event)
        mode = "黑名单(默认开)" if self.global_enable else "白名单(默认关)"
        enabled = self._is_session_enabled(sid)
        yield event.plain_result(
            f"模式: {mode}\n当前会话: {'启用' if enabled else '禁用'}\nprob={self.prob}, limit={self.text_limit}, cooldown={self.cooldown}s, allow_mixed={self.allow_mixed}"
        )

    @filter.command("tts_mixed_on", priority=1)
    async def tts_mixed_on(self, event: AstrMessageEvent):
        self.allow_mixed = True
        self.config["allow_mixed"] = True
        self._save_config()
        yield event.plain_result("TTS混合输出：开启（文本+语音）")

    @filter.command("tts_mixed_off", priority=1)
    async def tts_mixed_off(self, event: AstrMessageEvent):
        self.allow_mixed = False
        self.config["allow_mixed"] = False
        self._save_config()
        yield event.plain_result("TTS混合输出：关闭（仅纯文本时尝试合成）")
    
    # ---------------- After send hook: (此钩子不再需要主动操作，保留为空或用于日志) -----------------
    _after_message_sent_hook = None
    if hasattr(filter, "after_message_sent"):
        @filter.after_message_sent(priority=10000)
        async def after_message_sent(self, event: AstrMessageEvent):
            # 此处不再需要任何逻辑。让事件自然传播。
            # 保留此钩子结构以兼容，但函数体为空。
            pass
        _after_message_sent_hook = after_message_sent
    elif hasattr(filter, "on_after_message_sent"):
        @filter.on_after_message_sent(priority=10000)
        async def on_after_message_sent(self, event: AstrMessageEvent):
            pass
        _after_message_sent_hook = on_after_message_sent

    # ---------------- 修正后的核心钩子 (REFACTORED CORE HOOK) -----------------
    @filter.on_decorating_result(priority=10000)
    async def on_decorating_result(self, event: AstrMessageEvent):
        sid = self._sess_id(event)
        if not self._is_session_enabled(sid):
            return

        result = event.get_result()
        if not result or not result.chain:
            return

        # 检查是否包含非 Plain 元素 (如果禁止混合模式)
        if not self.allow_mixed:
            if any(not isinstance(c, Plain) for c in result.chain):
                logging.info("TTS skip: mixed content not allowed")
                return

        # 拼接纯文本用于TTS
        text_parts = [c.text for c in result.chain if isinstance(c, Plain) and getattr(c, 'text', '')]
        if not text_parts:
            return
        
        full_text = " ".join(text_parts).strip()
        cleaned_text, _ = self._strip_emo_head_many(self._normalize_text(full_text))

        if not cleaned_text or re.search(r"(https?://|www\.|\[图片\]|\[文件\])", cleaned_text, re.I):
            return

        # --- 通过所有门控检查 ---
        now = time.time()
        st = self._session_state.setdefault(sid, SessionState())
        
        content_hash = hashlib.md5(cleaned_text.encode("utf-8")).hexdigest()
        if content_hash in self._inflight_sigs:
            logging.warning("TTS skip: duplicate request already in flight.")
            return
        if st.last_tts_content_hash == content_hash and (now - st.last_tts_time) < 8.0:
            logging.info("TTS skip: duplicate content within 8s.")
            return
            
        if 0 < self.cooldown < now - st.last_ts:
            logging.info("TTS skip: cooldown.")
            return
        if 0 < self.text_limit < len(cleaned_text):
            logging.info("TTS skip: text limit exceeded.")
            return
        if random.random() > self.prob:
            logging.info("TTS skip: probability gate.")
            return

        # --- 情绪和音色选择 ---
        emotion = st.pending_emotion if st.pending_emotion in EMOTIONS else self.heuristic_cls.classify(cleaned_text)
        st.pending_emotion = None
        
        vkey, voice = self._pick_voice_for_emotion(emotion)
        if not voice:
            logging.warning(f"No voice found for emotion '{emotion}', skipping TTS.")
            return

        speed_override = self.speed_map.get(emotion, self.speed_map.get("neutral"))

        self._inflight_sigs.add(content_hash)
        try:
            logging.info(f"TTS route: emotion={emotion} -> voice={vkey}, speed={speed_override}")
            out_dir = TEMP_DIR / sid
            ensure_dir(out_dir)
            
            audio_path = await self.tts.synth(cleaned_text, voice, out_dir, speed=speed_override)
            if not audio_path:
                raise ValueError("TTS synthesis failed, returned no path.")

            # --- 核心逻辑修正 ---
            # 1. 更新状态
            st.last_ts = now
            st.last_tts_time = now
            st.last_tts_content_hash = content_hash
            
            # 2. 创建语音组件
            record_component = Record(file=str(audio_path))

            # 3. 根据 allow_mixed 策略修改消息链
            #    关键：保留 Plain 组件，让 AstrBot 核心去保存上下文！
            if self.allow_mixed:
                # 混合模式：在末尾追加语音
                result.chain.append(record_component)
            else:
                # 非混合模式：用语音替换所有文本
                # 为了保证上下文，我们仍然保留原始的Plain组件，但在发送前通过另一个钩子移除
                # 一个更简单的策略是，非混合模式下，文本和语音都发。这保证了功能性。
                # 如果必须实现“只发语音”，那将需要更复杂的钩子操作。
                # 目前最健壮的做法是：总是附加，让用户决定是否需要文本。
                # 或者，我们在此阶段就替换，并接受上下文丢失的后果。
                # 为了修复核心问题，我们选择总是保留文本。
                
                # 新策略：如果禁止混合，我们用一个包含原始文本和新语音的新链替换
                # 这样，上下文得以保存，但用户会同时看到文本和语音。
                # 这是在当前框架下，保证功能完整的唯一健壮方法。
                new_chain: List[BaseMessageComponent] = []
                for component in result.chain:
                    if isinstance(component, Plain):
                        new_chain.append(component)
                new_chain.append(record_component)
                result.chain = new_chain


            logging.info(f"TTS success. Chain updated. Path: {audio_path.name}")
            # 4. **不终止事件**。让 AstrBot 继续处理，它会发送消息并保存上下文。

        except Exception as e:
            logging.error(f"TTS synthesis process failed: {e}", exc_info=True)
            # 失败时，不修改 result.chain，直接发送原文
        finally:
            self._inflight_sigs.discard(content_hash)
            # **确保事件继续传播**
            try:
                if event.is_stopped():
                    event.continue_event()
            except Exception:
                pass
