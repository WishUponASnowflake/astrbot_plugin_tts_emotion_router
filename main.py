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
    last_tts_sig: Optional[str] = None     # 上一次发送音频的签名
    last_tts_time: float = 0.0             # 上一次发送音频的时间戳


@register("astrabot_plugin_tts_emotion_router", "木有知", "按情绪路由到不同音色的TTS插件", "2.1.0")
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
        api_url = api.get("url", "")
        api_key = api.get("key", "")
        api_model = api.get("model", "gpt-tts-pro")
        api_format = api.get("format", "mp3")  # 默认 mp3，减少部分平台播放噪点
        api_speed = float(api.get("speed", 1.0))
        api_gain = float(api.get("gain", 5.0))   # +50% 增益
        api_sr = int(api.get("sample_rate", 44100 if api_format in ("mp3", "wav") else 48000))
        # 初始化 TTS 客户端（支持 gain 与 sample_rate）
        self.tts = SiliconFlowTTS(api_url, api_key, api_model, api_format, api_speed, gain=api_gain, sample_rate=api_sr)

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
        self.emo_marker_enable: bool = bool(marker_cfg.get("enable", True))  # 默认开启
        self.emo_marker_tag: str = str(marker_cfg.get("tag", "EMO"))
        try:
            tag = re.escape(self.emo_marker_tag)
            self._emo_marker_re = re.compile(rf"\[\s*{tag}\s*:\s*(happy|sad|angry|neutral)\s*\]", re.I)
        except Exception:
            self._emo_marker_re = None
        # 额外：更宽松的去除规则（允许 [EMO] / [EMO:] / 全角【EMO】 以及纯单词 emo 开头等变体）
        try:
            tag = re.escape(self.emo_marker_tag)
            # 允许“:[label]”可缺省label，接受半/全角冒号及连字符，锚定开头以仅清理头部
            self._emo_marker_re_any = re.compile(
                rf"^[\s\ufeff]*[\[\(【]\s*{tag}\s*(?:[:：-]\s*[a-z]*)?\s*[\]\)】]",
                re.I,
            )
            # 头部 token：支持 [EMO] / [EMO:] / 【EMO：】 / emo / emo:happy / 等，label 可缺省（限定四选一）
            self._emo_head_token_re = re.compile(
                rf"^[\s\ufeff]*(?:[\[\(【]\s*{tag}\s*(?:[:：-]\s*(?P<lbl>happy|sad|angry|neutral))?\s*[\]\)】]|(?:{tag}|emo)\s*(?:[:：-]\s*(?P<lbl2>happy|sad|angry|neutral))?)\s*[,，。:：-]*\s*",
                re.I,
            )
            # 头部 token（英文任意标签）：如 [EMO:confused]，先取 raw 再做同义词归一化
            self._emo_head_anylabel_re = re.compile(
                rf"^[\s\ufeff]*[\[\(【]\s*{tag}\s*[:：-]\s*(?P<raw>[a-z]+)\s*[\]\)】]",
                re.I,
            )
        except Exception:
            self._emo_marker_re_any = None
            self._emo_head_token_re = None
            self._emo_head_anylabel_re = None
        
        self._session_state: Dict[str, SessionState] = {}
        ensure_dir(TEMP_DIR)
        cleanup_dir(TEMP_DIR, ttl_seconds=6*3600)
        
        # 简单关键词启发，用于无标记时的中性偏置判定
        try:
            self._emo_kw = {
                "happy": re.compile(r"(开心|快乐|高兴|喜悦|愉快|兴奋|喜欢|令人开心|挺好|不错|开心|happy|joy|delight|excited|great|awesome|lol)", re.I),
                "sad": re.compile(r"(伤心|难过|沮丧|低落|悲伤|哭|流泪|难受|失望|委屈|心碎|sad|depress|upset|unhappy|blue|tear)", re.I),
                "angry": re.compile(r"(生气|愤怒|火大|恼火|气愤|气死|怒|怒了|生气了|angry|furious|mad|rage|annoyed|irritat)", re.I),
            }
        except Exception:
            self._emo_kw = {
                "happy": re.compile(r"happy|joy|delight|excited", re.I),
                "sad": re.compile(r"sad|depress|upset|unhappy", re.I),
                "angry": re.compile(r"angry|furious|mad|rage", re.I),
            }
        
        # 去重机制 - 简化版
        self._event_guard: Dict[str, float] = {}

    # ---------------- helpers -----------------
    def _event_key(self, event: AstrMessageEvent) -> Optional[str]:
        """构造事件唯一键，用于避免同一消息事件重复处理。"""
        try:
            gid = ""
            try:
                gid = event.get_group_id() or ""
            except Exception:
                gid = ""
            sid = ""
            try:
                sid = event.get_sender_id() or ""
            except Exception:
                sid = ""
            mid = None
            for attr in ("get_message_id", "get_msg_id", "message_id", "msg_id"):
                try:
                    v = getattr(event, attr)
                    mid = v() if callable(v) else v
                    if mid:
                        break
                except Exception:
                    continue
            base = f"{gid}|{sid}|{mid or ''}"
            if base.strip("|"):
                return base
        except Exception:
            pass
        return None

    def _sender_key(self, event: AstrMessageEvent) -> Optional[str]:
        """构造"群ID|发送者ID"的键，不含消息ID，用于同一发送者短时间重复触发的去重。"""
        try:
            gid = ""
            try:
                gid = event.get_group_id() or ""
            except Exception:
                gid = ""
            sid = ""
            try:
                sid = event.get_sender_id() or ""
            except Exception:
                sid = ""
            base = f"{gid}|{sid}"
            if base.strip("|"):
                return base
        except Exception:
            pass
        return None

    def _sweep_event_guard(self):
        """清理过期的事件去重记录。"""
        try:
            now = time.time()
            for k, ts in list(self._event_guard.items()):
                if now - ts > 60:
                    self._event_guard.pop(k, None)
        except Exception:
            pass

    def _strip_leading_mentions(self, text: str) -> str:
        """去掉开头的 @提及 / 回复 @某人 前缀，避免因仅有提及差异而绕过去重或读出无意义内容。"""
        if not text:
            return text
        try:
            # 连续剥离多重前缀，如："回复 @张三: 回复 @李四， ..."
            while True:
                t = text
                # 形式1：回复 @xxx: / 回复@xxx，/ Re @xxx:
                t = re.sub(r"^\s*(?:回复|Re|RE)\s*@[^\s:：,，-]+\s*[:：,，-]*\s*", "", t)
                # 形式2：@xxx: / @xxx： / @xxx，
                t = re.sub(r"^\s*@[^\s:：,，-]+\s*[:：,，-]*\s*", "", t)
                if t == text:
                    break
                text = t
        except Exception:
            pass
        return text

    def _build_tts_sig(self, text: str, vkey: Optional[str], voice: Optional[str], speed: Optional[float]) -> str:
        """构造一次TTS请求的签名，用于短时间去重。"""
        try:
            import hashlib
            payload = {
                "t": (text or "")[:200],  # 取前200字符足够判等
                "vk": vkey or "",
                "v": voice or "",
                "s": None if speed is None else float(speed),
            }
            s = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            return hashlib.sha1(s.encode("utf-8")).hexdigest()
        except Exception:
            return f"fallback:{(text or '')[:60]}:{vkey}:{speed}"

    def _build_text_sig(self, text: str) -> str:
        """对文本做宽松规范化后计算签名：
        - 去不可见字符与BOM
        - 去掉开头@/回复@前缀
        - 合并空白、去常见标点、转小写
        - 截取前200字符再hash
        """
        try:
            import hashlib
            t = self._normalize_text(text or "")
            t = self._strip_leading_mentions(t)
            t = re.sub(r"\s+", " ", t).strip().lower()
            # 去常见中英文标点，避免仅标点差异造成重复
            t = re.sub(r"[，。,.!？?…~:：;；\-—\"\"\"'''()（）\[\]【】<>《》@#]", "", t)
            t = t[:200]
            return hashlib.sha1(t.encode("utf-8")).hexdigest()
        except Exception:
            return f"txtsig:{(text or '')[:60]}"

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

    def _normalize_text(self, text: str) -> str:
        """移除不可见字符与BOM，避免破坏头部匹配。"""
        if not text:
            return text
        invisibles = [
            "\ufeff",  # BOM
            "\u200b", "\u200c", "\u200d", "\u200e", "\u200f",  # ZW* & RTL/LTR marks
            "\u202a", "\u202b", "\u202c", "\u202d", "\u202e",  # directional marks
        ]
        for ch in invisibles:
            text = text.replace(ch, "")
        return text

    def _normalize_label(self, label: Optional[str]) -> Optional[str]:
        """将任意英文/中文情绪词映射到四选一。
        例：confused->neutral，upset->sad，furious->angry，delighted->happy 等。"""
        if not label:
            return None
        l = label.strip().lower()
        mapping = {
            "happy": {
                "happy", "joy", "joyful", "cheerful", "delighted", "excited", "smile", "positive",
                "开心", "快乐", "高兴", "喜悦", "兴奋", "愉快",
            },
            "sad": {
                "sad", "sorrow", "sorrowful", "depressed", "down", "unhappy", "cry", "crying", "tearful", "blue", "upset",
                "伤心", "难过", "沮丧", "低落", "悲伤", "流泪",
            },
            "angry": {
                "angry", "mad", "furious", "annoyed", "irritated", "rage", "rageful", "wrath",
                "生气", "愤怒", "恼火", "气愤",
            },
            "neutral": {
                "neutral", "calm", "plain", "normal", "objective", "ok", "fine", "meh", "average", "confused", "uncertain", "unsure",
                "平静", "冷静", "一般", "中立", "客观", "困惑", "迷茫",
            },
        }
        for k, vs in mapping.items():
            if l in vs:
                return k
        return None

    def _pick_voice_for_emotion(self, emotion: str):
        """根据情绪选择音色：优先 exact -> neutral -> 偏好映射 -> 任意非空。
        返回 (voice_key, voice_uri)；若无可用则 (None, None)。"""
        vm = self.voice_map or {}
        # exact
        v = vm.get(emotion)
        if v:
            return emotion, v
        # neutral
        v = vm.get("neutral")
        if v:
            return "neutral", v
        # 偏好映射（让缺失的项落到最接近的可用音色）
        pref = {"sad": "angry", "angry": "angry", "happy": "happy", "neutral": "happy"}
        for key in [pref.get(emotion), "happy", "angry"]:
            if key and vm.get(key):
                return key, vm[key]
        # 兜底：任意非空
        for k, v in vm.items():
            if v:
                return k, v
        return None, None

    def _strip_emo_head(self, text: str) -> tuple[str, Optional[str]]:
        """从文本开头剥离各种 EMO/emo 标记变体，并返回(清理后的文本, 解析到的情绪或None)。"""
        if not text:
            return text, None
        # 优先用宽松的头部匹配（限定四选一）
        if self._emo_head_token_re:
            m = self._emo_head_token_re.match(text)
            if m:
                label = (m.group('lbl') or m.group('lbl2') or "").lower()
                if label not in EMOTIONS:
                    label = None
                cleaned = self._emo_head_token_re.sub("", text, count=1)
                return cleaned.strip(), label
        # 其次：捕获任意英文标签，再做同义词归一化
        if self._emo_head_anylabel_re:
            m2 = self._emo_head_anylabel_re.match(text)
            if m2:
                raw = (m2.group('raw') or "").lower()
                label = self._normalize_label(raw)
                cleaned = self._emo_head_anylabel_re.sub("", text, count=1)
                return cleaned.strip(), label
        # 最后：去掉任何形态头部标签（即便无法识别标签含义也移除）
        if self._emo_marker_re_any and text.lstrip().startswith(("[", "【", "(")):
            cleaned = self._emo_marker_re_any.sub("", text, count=1)
            return cleaned.strip(), None
        return text, None

    def _strip_emo_head_many(self, text: str) -> tuple[str, Optional[str]]:
        """连续剥离多枚开头的EMO/emo标记（若LLM/其它插件重复注入）。返回(清理后文本, 最后一次解析到的情绪)。"""
        last_label: Optional[str] = None
        while True:
            cleaned, label = self._strip_emo_head(text)
            if label:
                last_label = label
            if cleaned == text:
                break
            text = cleaned
        return text, last_label

    # ---------------- LLM 请求前：注入情绪标记指令 -----------------
    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, request):
        """在系统提示中加入隐藏情绪标记指令，让 LLM 先输出 [EMO:xxx] 再回答。"""
        if not self.emo_marker_enable:
            return
        try:
            tag = self.emo_marker_tag
            instr = (
                f"请在每次回复的最开头只输出一个隐藏情绪标记，格式严格为："
                f"[{tag}:happy] 或 [{tag}:sad] 或 [{tag}:angry] 或 [{tag}:neutral]。"
                "必须四选一；若无法判断请选择 neutral。该标记仅供系统解析，"
                "输出后立刻继续正常作答，不要解释或复述该标记。"
                "如你想到其它词，请映射到以上四类：happy(开心/喜悦/兴奋)、sad(伤心/难过/沮丧/upset)、"
                "angry(生气/愤怒/恼火/furious)、neutral(平静/普通/困惑/confused)。"
            )
            request.system_prompt = (request.system_prompt or "") + "\n" + instr
        except Exception:
            pass

    # ---------------- LLM 标记解析（避免标签外显） -----------------
    @filter.on_llm_response(priority=1)
    async def on_llm_response(self, event: AstrMessageEvent, response: LLMResponse):
        if not self.emo_marker_enable:
            return
        label: Optional[str] = None

        # 1) 尝试从 completion_text 提取并清理
        try:
            text = getattr(response, "completion_text", None)
            if isinstance(text, str) and text.strip():
                t0 = self._normalize_text(text)
                cleaned, l1 = self._strip_emo_head_many(t0)
                if l1 in EMOTIONS:
                    label = l1
                response.completion_text = cleaned
        except Exception:
            pass

        # 2) 无论 completion_text 是否为空，都从 result_chain 首个 Plain 再尝试一次
        try:
            rc = getattr(response, "result_chain", None)
            if rc and hasattr(rc, "chain") and rc.chain:
                new_chain = []
                cleaned_once = False
                for comp in rc.chain:
                    if not cleaned_once and isinstance(comp, Plain) and getattr(comp, "text", None):
                        t0 = self._normalize_text(comp.text)
                        t, l2 = self._strip_emo_head_many(t0)
                        if l2 in EMOTIONS and label is None:
                            label = l2
                        if t:
                            new_chain.append(Plain(text=t))
                        cleaned_once = True
                    else:
                        new_chain.append(comp)
                rc.chain = new_chain
        except Exception:
            pass

        # 3) 记录到 session
        if label in EMOTIONS:
            sid = self._sess_id(event)
            st = self._session_state.setdefault(sid, SessionState())
            st.pending_emotion = label

    # ---------------- Commands -----------------
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
        """手动指定下一条消息的情绪用于路由：tts_emote happy|sad|angry|neutral"""
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

    @filter.command("tts_gain", priority=1)
    async def tts_gain(self, event: AstrMessageEvent, *, value: Optional[str] = None):
        """调节输出音量增益（单位dB，范围 -10 ~ 10）。示例：tts_gain 5"""
        try:
            if value is None:
                raise ValueError
            v = float(value)
            assert -10.0 <= v <= 10.0
            # 更新运行期
            try:
                self.tts.gain = v
            except Exception:
                pass
            # 持久化
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
        yield event.plain_result(f"模式: {mode}\n当前会话: {'启用' if enabled else '禁用'}\nprob={self.prob}, limit={self.text_limit}, cooldown={self.cooldown}s")

    # ---------------- Core hook -----------------
    @filter.on_decorating_result()
    async def on_decorating_result(self, event: AstrMessageEvent):
        # 简化的事件去重：同一消息事件5s内只处理一次
        ek = None
        try:
            ek = self._event_key(event)
            if ek:
                self._sweep_event_guard()
                last = self._event_guard.get(ek)
                if last and (time.time() - last) < 5:
                    logging.debug("TTS: skip duplicate event within 5s")
                    return
                # 先占位，避免并发重复进入
                self._event_guard[ek] = time.time()
        except Exception:
            pass

        sid = self._sess_id(event)
        if not self._is_session_enabled(sid):
            return

        # 结果链
        result = event.get_result()
        if not result or not result.chain:
            return

        # 在最终输出层面，仅对首个 Plain 的开头执行一次剥离，确保不会把“emo”读出来
        try:
            new_chain = []
            cleaned_once = False
            for comp in result.chain:
                if not cleaned_once and isinstance(comp, Plain) and getattr(comp, "text", None):
                    t0 = comp.text
                    t0 = self._normalize_text(t0)
                    t, _ = self._strip_emo_head_many(t0)
                    # 去掉开头的 @提及/回复@某人
                    t = self._strip_leading_mentions(t)
                    if t:
                        new_chain.append(Plain(text=t))
                    cleaned_once = True
                else:
                    new_chain.append(comp)
            result.chain = new_chain
        except Exception:
            pass

        # 是否允许混合
        if not self.allow_mixed and any(not isinstance(c, Plain) for c in result.chain):
            return

        # 拼接纯文本
        text_parts = [c.text.strip() for c in result.chain if isinstance(c, Plain) and c.text.strip()]
        if not text_parts:
            return
        text = " ".join(text_parts)
        
        # 基础文本检查
        if not text or not text.strip():
            return

        # 归一化 + 连续剥离（终极兜底）
        orig_text = text
        text = self._normalize_text(text)
        text, _ = self._strip_emo_head_many(text)
        # TTS 之前移除开头提及
        text = self._strip_leading_mentions(text)
        
        # 清理后再次检查文本是否有效
        if not text or not text.strip():
            return

        # 过滤链接/文件等提示性内容，避免朗读
        if re.search(r"(https?://|www\.|\[图片\]|\[文件\]|\[转发\]|\[引用\]|\.jpg|\.png|\.gif|\.jpeg|\.webp|\.mp4|\.avi|\.mov|\.pdf|\.doc|\.txt|\.zip|\.rar)", text, re.I):
            return
            
        # 过滤纯文件扩展名或路径或单个词
        if re.match(r"^\s*(\.(jpg|png|gif|jpeg|webp|mp4|avi|mov|pdf|doc|txt|zip|rar)|jpg|png|gif|jpeg|webp)\s*$", text, re.I):
            return
            
        # 过滤太短的文本（可能是无意义的输出）
        if len(text.strip()) < 2:
            return
            
        # 过滤纯符号或数字
        if re.match(r"^\s*[^\w\u4e00-\u9fff]*\s*$", text) or re.match(r"^\s*\d+\s*$", text):
            return

        # 随机/冷却/长度
        st = self._session_state.setdefault(sid, SessionState())
        now = time.time()
        if self.cooldown > 0 and (now - st.last_ts) < self.cooldown:
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
            src = "tag"
        else:
            emotion = self.heuristic_cls.classify(text, context=None)
            src = "heuristic"
            # 中性偏置：若文本无明显情绪关键词，则强制使用 neutral
            try:
                kw = getattr(self, "_emo_kw", {})
                has_kw = any(p.search(text) for p in kw.values())
                if not has_kw:
                    emotion = "neutral"
            except Exception:
                pass
        
        vkey, voice = self._pick_voice_for_emotion(emotion)
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

        # 简单的TTS去重：相同文本内容3秒内只生成一次
        tts_sig = self._build_tts_sig(text, vkey, voice, speed_override)
        now = time.time()
        if st.last_tts_sig == tts_sig and (now - st.last_tts_time) < 3:
            logging.debug("TTS: skip duplicate TTS within 3s")
            return

        logging.info(
            "TTS route: emotion=%s(src=%s) -> %s (%s), speed=%s",
            emotion,
            src,
            vkey,
            (voice[:40] + "...") if isinstance(voice, str) and len(voice) > 43 else voice,
            speed_override if speed_override is not None else getattr(self.tts, "speed", None),
        )
        logging.debug("TTS input head(before/after): %r -> %r", orig_text[:60], text[:60])

        out_dir = TEMP_DIR / sid
        ensure_dir(out_dir)
        # 最后一重防线：若 TTS 前文本仍以 emo/token 开头，强制清理
        try:
            if text and (text.lower().lstrip().startswith("emo") or text.lstrip().startswith(("[", "【", "("))):
                text, _ = self._strip_emo_head_many(text)
        except Exception:
            pass
        audio_path = self.tts.synth(text, voice, out_dir, speed=speed_override)
        if not audio_path:
            logging.error("TTS调用失败，降级为文本")
            return

        st.last_ts = time.time()
        st.last_tts_sig = tts_sig
        st.last_tts_time = st.last_ts
        result.chain = [Record(file=str(audio_path))]
