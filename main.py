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
import asyncio

def _ensure_compatible_astrbot():
    """确保 astrbot API 兼容；若宿主astrbot不满足需要，回退到插件自带的 AstrBot。"""
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

# 兼容不同 AstrBot 版本的导入：event 可能是模块(event.py)也可能是包(event/)
try:
    # 优先常规路径
    from astrbot.api.event import AstrMessageEvent  # type: ignore
except Exception:  # pragma: no cover - 旧版本回退
    from astrbot.core.platform import AstrMessageEvent  # type: ignore

# 统一获取 filter 装饰器集合：
try:
    # 新版通常支持 from astrbot.api.event import filter
    from astrbot.api.event import filter as filter  # type: ignore
except Exception:
    try:
        # 另一些版本可 import 子模块
        import importlib as _importlib
        filter = _importlib.import_module("astrbot.api.event.filter")  # type: ignore
    except Exception:
        # 最后回退：用 register 构造一个拥有同名方法的轻量代理
        try:
            import astrbot.core.star.register as _reg  # type: ignore

            class _FilterCompat:
                def command(self, *a, **k):
                    return _reg.register_command(*a, **k)

                def on_llm_request(self, *a, **k):
                    return _reg.register_on_llm_request(*a, **k)

                def on_llm_response(self, *a, **k):
                    return _reg.register_on_llm_response(*a, **k)

                def on_decorating_result(self, *a, **k):
                    return _reg.register_on_decorating_result(*a, **k)

                def after_message_sent(self, *a, **k):
                    return _reg.register_after_message_sent(*a, **k)

                # 兼容某些版本名为 on_after_message_sent
                def on_after_message_sent(self, *a, **k):
                    return _reg.register_after_message_sent(*a, **k)

            filter = _FilterCompat()  # type: ignore
        except Exception as _e:  # 若三种方式均失败，抛出原错误
            raise _e
from astrbot.api.star import Context, Star, register
# 优先使用 core 版本的组件类型以匹配 RespondStage 校验逻辑，失败时回退到 api 版本
try:  # pragma: no cover - 运行期按宿主 AstrBot 能力选择
    from astrbot.core.message.components import Record, Plain  # type: ignore
except Exception:  # pragma: no cover - 旧版本回退
    from astrbot.api.message_components import Record, Plain  # type: ignore
from astrbot.core.config.astrbot_config import AstrBotConfig
from astrbot.api.provider import LLMResponse
from astrbot.core.message.message_event_result import ResultContentType

from .emotion.infer import EMOTIONS
from .emotion.classifier import HeuristicClassifier  # LLMClassifier 不再使用
from .tts.provider_siliconflow import SiliconFlowTTS
from .utils.audio import ensure_dir, cleanup_dir
from .utils.extract import extractor, ExtractedContent

# 记录 astrbot 实际来源，便于远端排查“导入到插件内自带 AstrBot”的问题
try:
    import astrbot as _ab_mod  # type: ignore
    logging.info("TTSEmotionRouter: using astrbot from %s", getattr(_ab_mod, "__file__", None))
except Exception:
    pass

CONFIG_FILE = Path(__file__).parent / "config.json"  # 旧版本地文件，作为迁移来源
TEMP_DIR = Path(__file__).parent / "temp"


@dataclass
class SessionState:
    last_ts: float = 0.0
    pending_emotion: Optional[str] = None  # 基于隐藏标记的待用情绪
    last_tts_content: Optional[str] = None  # 最后生成的TTS内容（防重复）
    last_tts_time: float = 0.0  # 最后TTS生成时间
    last_assistant_text: Optional[str] = None  # 最近一次助手可读文本（用于兜底入库）
    last_assistant_text_time: float = 0.0


@register(
    "astrbot_plugin_tts_emotion_router",
    "木有知",
    "按情绪路由到不同音色的TTS插件",
    "0.3.0",
)
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
                        "global_enable",
                        "enabled_sessions",
                        "disabled_sessions",
                        "prob",
                        "text_limit",
                        "cooldown",
                        "allow_mixed",
                        "api",
                        "voice_map",
                        "emotion",
                        "speed_map",
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
        api_gain = float(api.get("gain", 5.0))  # +50% 增益
        api_sr = int(
            api.get("sample_rate", 44100 if api_format in ("mp3", "wav") else 48000)
        )
        # 初始化 TTS 客户端（支持 gain 与 sample_rate）
        self.tts = SiliconFlowTTS(
            api_url,
            api_key,
            api_model,
            api_format,
            api_speed,
            gain=api_gain,
            sample_rate=api_sr,
        )

        self.voice_map: Dict[str, str] = self.config.get("voice_map", {})
        self.speed_map: Dict[str, float] = self.config.get("speed_map", {}) or {}
        self.global_enable: bool = bool(self.config.get("global_enable", True))
        self.enabled_sessions: List[str] = list(self.config.get("enabled_sessions", []))
        self.disabled_sessions: List[str] = list(
            self.config.get("disabled_sessions", [])
        )
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
            self._emo_marker_re = re.compile(
                rf"\[\s*{tag}\s*:\s*(happy|sad|angry|neutral)\s*\]", re.I
            )
        except Exception:
            self._emo_marker_re = None
        # 额外：更宽松的去除规则（允许 [EMO] / [EMO:] / 全角【EMO】 以及纯单词 emo 开头等变体）
        try:
            tag = re.escape(self.emo_marker_tag)
            # 允许“:[label]”可缺省label，接受半/全角冒号及连字符，锚定开头以仅清理头部
            self._emo_marker_re_any = re.compile(
                rf"^[\s\ufeff]*[\[\(【]\s*{tag}\s*(?:[:\uff1a-]\s*[a-z]*)?\s*[\]\)】]",
                re.I,
            )
            # 头部 token：支持 [EMO] / [EMO:] / 【EMO：】 / emo / emo:happy / 等，label 可缺省（限定四选一）
            self._emo_head_token_re = re.compile(
                rf"^[\s\ufeff]*(?:[\[\(【]\s*{tag}\s*(?:[:\uff1a-]\s*(?P<lbl>happy|sad|angry|neutral))?\s*[\]\)】]|(?:{tag}|emo)\s*(?:[:\uff1a-]\s*(?P<lbl2>happy|sad|angry|neutral))?)\s*[,，。:\uff1a-]*\s*",
                re.I,
            )
            # 头部 token（英文任意标签）：如 [EMO:confused]，先取 raw 再做同义词归一化
            self._emo_head_anylabel_re = re.compile(
                rf"^[\s\ufeff]*[\[\(【]\s*{tag}\s*[:\uff1a-]\s*(?P<raw>[a-z]+)\s*[\]\)】]",
                re.I,
            )
        except Exception:
            self._emo_marker_re_any = None
            self._emo_head_token_re = None
            self._emo_head_anylabel_re = None

        self._session_state: Dict[str, SessionState] = {}
        # 事件级防重：最近发送签名与进行中签名
        self._recent_sends: Dict[str, float] = {}
        self._inflight_sigs: set[str] = set()
        ensure_dir(TEMP_DIR)
        # 初始清理：删除超过2小时的文件
        cleanup_dir(TEMP_DIR, ttl_seconds=2 * 3600)

        # 简单关键词启发，用于无标记时的中性偏置判定
        try:
            self._emo_kw = {
                "happy": re.compile(
                    r"(开心|快乐|高兴|喜悦|愉快|兴奋|喜欢|令人开心|挺好|不错|开心|happy|joy|delight|excited|great|awesome|lol)",
                    re.I,
                ),
                "sad": re.compile(
                    r"(伤心|难过|沮丧|低落|悲伤|哭|流泪|难受|失望|委屈|心碎|sad|depress|upset|unhappy|blue|tear)",
                    re.I,
                ),
                "angry": re.compile(
                    r"(生气|愤怒|火大|恼火|气愤|气死|怒|怒了|生气了|angry|furious|mad|rage|annoyed|irritat)",
                    re.I,
                ),
            }
        except Exception:
            self._emo_kw = {
                "happy": re.compile(r"happy|joy|delight|excited", re.I),
                "sad": re.compile(r"sad|depress|upset|unhappy", re.I),
                "angry": re.compile(r"angry|furious|mad|rage", re.I),
            }

    def _is_our_record(self, comp) -> bool:
        try:
            if not isinstance(comp, Record):
                return False
            f = getattr(comp, "file", "") or ""
            if not f:
                return False
            fpath = Path(f)
            return str(fpath).startswith(str((Path(__file__).parent / "temp").resolve()))
        except Exception:
            return False

    def _validate_audio_file(self, audio_path: Path) -> bool:
        """验证音频文件是否有效"""
        try:
            if not audio_path.exists():
                logging.error(f"TTSEmotionRouter: 音频文件不存在: {audio_path}")
                return False
            
            file_size = audio_path.stat().st_size
            if file_size == 0:
                logging.error(f"TTSEmotionRouter: 音频文件为空: {audio_path}")
                return False
            
            if file_size < 100:  # 小于100字节通常是无效文件
                logging.error(f"TTSEmotionRouter: 音频文件太小({file_size}字节): {audio_path}")
                return False
            
            # 检查文件扩展名
            if audio_path.suffix.lower() not in ['.mp3', '.wav', '.opus', '.pcm']:
                logging.warning(f"TTSEmotionRouter: 音频文件格式可能不支持: {audio_path}")
            
            logging.info(f"TTSEmotionRouter: 音频文件验证通过: {audio_path} ({file_size}字节)")
            return True
        except Exception as e:
            logging.error(f"TTSEmotionRouter: 音频文件验证失败: {audio_path}, 错误: {e}")
            return False

    def _normalize_audio_path(self, audio_path: Path) -> str:
        """规范化音频文件路径以提高协议端兼容性"""
        try:
            # 1. 确保使用绝对路径
            abs_path = audio_path.resolve()
            
            # 2. Windows路径格式转换
            import os
            normalized = os.path.normpath(str(abs_path))
            
            # 3. 对于某些协议端，可能需要使用正斜杠
            if os.name == 'nt':  # Windows
                # 先尝试使用反斜杠路径（标准Windows格式）
                return normalized
            else:
                # Unix-like系统使用正斜杠
                return normalized.replace('\\', '/')
        except Exception as e:
            logging.error(f"TTSEmotionRouter: 路径规范化失败: {audio_path}, 错误: {e}")
            return str(audio_path)

    def _create_fallback_text_result(self, text: str, event: AstrMessageEvent) -> None:
        """创建文本回退结果"""
        try:
            result = event.get_result()
            if result and hasattr(result, 'chain'):
                # 清空现有链并添加文本结果
                result.chain.clear()
                result.chain.append(Plain(text))
                logging.info(f"TTSEmotionRouter: 已回退到文本消息: {text[:50]}...")
        except Exception as e:
            logging.error(f"TTSEmotionRouter: 创建文本回退失败: {e}")

    def _try_copy_to_accessible_location(self, audio_path: Path) -> Optional[Path]:
        """尝试将音频文件复制到更容易访问的位置"""
        try:
            import tempfile
            import shutil
            
            # 使用系统临时目录
            temp_dir = Path(tempfile.gettempdir()) / "astrbot_audio"
            temp_dir.mkdir(exist_ok=True)
            
            # 生成新的文件名
            import uuid
            new_filename = f"tts_{uuid.uuid4().hex[:8]}{audio_path.suffix}"
            new_path = temp_dir / new_filename
            
            # 复制文件
            shutil.copy2(audio_path, new_path)
            
            if self._validate_audio_file(new_path):
                logging.info(f"TTSEmotionRouter: 音频文件已复制到: {new_path}")
                return new_path
            else:
                # 清理失败的复制
                try:
                    new_path.unlink()
                except:
                    pass
                return None
        except Exception as e:
            logging.error(f"TTSEmotionRouter: 复制音频文件失败: {e}")
            return None

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
            CONFIG_FILE.write_text(
                json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            pass
        return merged

    def _save_config(self):
        # 面板配置优先保存到 data/config/tts_emotion_router_config.json
        if isinstance(self.config, AstrBotConfig):
            self.config.save_config()
        else:
            try:
                CONFIG_FILE.write_text(
                    json.dumps(self.config, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
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
            "\u200b",
            "\u200c",
            "\u200d",
            "\u200e",
            "\u200f",  # ZW* & RTL/LTR marks
            "\u202a",
            "\u202b",
            "\u202c",
            "\u202d",
            "\u202e",  # directional marks
        ]
        for ch in invisibles:
            text = text.replace(ch, "")
        return text

    def _normalize_label(self, label: Optional[str]) -> Optional[str]:
        """将任意英文/中文情绪词映射到四选一。
        例：confused->neutral，upset->sad，furious->angry，delighted->happy 等。"""
        if not label:
            return None
        lbl = label.strip().lower()
        mapping = {
            "happy": {
                "happy",
                "joy",
                "joyful",
                "cheerful",
                "delighted",
                "excited",
                "smile",
                "positive",
                "开心",
                "快乐",
                "高兴",
                "喜悦",
                "兴奋",
                "愉快",
            },
            "sad": {
                "sad",
                "sorrow",
                "sorrowful",
                "depressed",
                "down",
                "unhappy",
                "cry",
                "crying",
                "tearful",
                "blue",
                "upset",
                "伤心",
                "难过",
                "沮丧",
                "低落",
                "悲伤",
                "流泪",
            },
            "angry": {
                "angry",
                "mad",
                "furious",
                "annoyed",
                "irritated",
                "rage",
                "rageful",
                "wrath",
                "生气",
                "愤怒",
                "恼火",
                "气愤",
            },
            "neutral": {
                "neutral",
                "calm",
                "plain",
                "normal",
                "objective",
                "ok",
                "fine",
                "meh",
                "average",
                "confused",
                "uncertain",
                "unsure",
                "平静",
                "冷静",
                "一般",
                "中立",
                "客观",
                "困惑",
                "迷茫",
            },
        }
        for k, vs in mapping.items():
            if lbl in vs:
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
                label = (m.group("lbl") or m.group("lbl2") or "").lower()
                if label not in EMOTIONS:
                    label = None
                cleaned = self._emo_head_token_re.sub("", text, count=1)
                return cleaned.strip(), label
        # 其次：捕获任意英文标签，再做同义词归一化
        if self._emo_head_anylabel_re:
            m2 = self._emo_head_anylabel_re.match(text)
            if m2:
                raw = (m2.group("raw") or "").lower()
                label = self._normalize_label(raw)
                cleaned = self._emo_head_anylabel_re.sub("", text, count=1)
                return cleaned.strip(), label
        # 最后：去掉任何形态头部标记（即便无法识别标签含义也移除）
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
            # 简要调试：记录上下文条数与本轮 prompt 长度，便于排查“上下文丢失”
            try:
                ctxs = getattr(request, "contexts", None)
                clen = len(ctxs) if isinstance(ctxs, list) else 0
                plen = len(getattr(request, "prompt", "") or "")
                logging.info(f"TTSEmotionRouter.on_llm_request: contexts={clen}, prompt_len={plen}")
            except Exception:
                pass
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
            # 避免重复注入：仅当当前 system_prompt/prompt 中没有我们的标签时注入
            sp = getattr(request, "system_prompt", "") or ""
            pp = getattr(request, "prompt", "") or ""
            marker_present = (self.emo_marker_tag in sp) or (self.emo_marker_tag in pp)
            if not marker_present:
                # 以更高优先级前置到 system_prompt 顶部
                try:
                    request.system_prompt = (instr + "\n" + sp).strip()
                except Exception:
                    pass
                # 同时在 prompt 顶部再前置一次，兼容部分来源只读取 prompt 的实现
                try:
                    request.prompt = (instr + "\n\n" + pp).strip()
                except Exception:
                    pass
                # 尝试向 contexts 注入一条 system 消息（弱依赖，失败忽略）
                try:
                    ctxs = getattr(request, "contexts", None)
                    if isinstance(ctxs, list):
                        # 插到最前，提升优先级
                        ctxs.insert(0, {"role": "system", "content": instr})
                        request.contexts = ctxs
                except Exception:
                    pass
            # 简要调试：记录上下文条数与本轮 prompt 长度，便于排查“上下文丢失”
            try:
                ctxs = getattr(request, "contexts", None)
                clen = len(ctxs) if isinstance(ctxs, list) else 0
                plen = len(getattr(request, "prompt", "") or "")
                logging.info(f"TTSEmotionRouter.on_llm_request: injected={not marker_present}, contexts={clen}, prompt_len={plen}")
            except Exception:
                pass
        except Exception:
            pass

    # ---------------- LLM 标记解析（避免标签外显） -----------------
    @filter.on_llm_response(priority=1)
    async def on_llm_response(self, event: AstrMessageEvent, response: LLMResponse):
        if not self.emo_marker_enable:
            return
        label: Optional[str] = None
        cached_text: Optional[str] = None

        # 1) 尝试从 completion_text 提取并清理
        try:
            text = getattr(response, "completion_text", None)
            if isinstance(text, str) and text.strip():
                t0 = self._normalize_text(text)
                cleaned, l1 = self._strip_emo_head_many(t0)
                if l1 in EMOTIONS:
                    label = l1
                response.completion_text = cleaned
                # 兼容某些 AstrBot 内部使用 _completion_text 的实现，显式同步私有字段
                try:
                    setattr(response, "_completion_text", cleaned)
                except Exception:
                    pass
                cached_text = cleaned or cached_text
        except Exception:
            pass

        # 2) 无论 completion_text 是否为空，都从 result_chain 首个 Plain 再尝试一次
        try:
            rc = getattr(response, "result_chain", None)
            if rc and hasattr(rc, "chain") and rc.chain:
                new_chain = []
                cleaned_once = False
                for comp in rc.chain:
                    if (
                        not cleaned_once
                        and isinstance(comp, Plain)
                        and getattr(comp, "text", None)
                    ):
                        t0 = self._normalize_text(comp.text)
                        t, l2 = self._strip_emo_head_many(t0)
                        if l2 in EMOTIONS and label is None:
                            label = l2
                        if t:
                            new_chain.append(Plain(text=t))
                            # 若 completion_text 为空，则用首个 Plain 的清洗文本回填到 _completion_text
                            try:
                                if t and not getattr(response, "_completion_text", None):
                                    setattr(response, "_completion_text", t)
                            except Exception:
                                pass
                            cached_text = t or cached_text
                        cleaned_once = True
                    else:
                        new_chain.append(comp)
                rc.chain = new_chain
        except Exception:
            pass

        # 3) 记录到 session
        try:
            sid = self._sess_id(event)
            st = self._session_state.setdefault(sid, SessionState())
            if label in EMOTIONS:
                st.pending_emotion = label
            # 缓存可读文本，供只剩下 Record 的兜底入库
            if cached_text and cached_text.strip():
                st.last_assistant_text = cached_text.strip()
                st.last_assistant_text_time = time.time()
        except Exception:
            pass

        # 4) 立即尝试将清洗后的文本写入会话历史（幂等），避免后续阶段被误判 STOP 时丢上下文
        try:
            if cached_text and cached_text.strip():
                ok = await self._append_assistant_text_to_history(event, cached_text.strip())
                # 若此刻会话尚未建立，延迟一次重试
                if not ok:
                    try:
                        asyncio.create_task(self._delayed_history_write(event, cached_text.strip(), delay=0.8))
                    except Exception:
                        pass
        except Exception:
            pass

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
        """
        手动指定下一条消息的情绪用于路由：tts_emote happy|sad|angry|neutral
        """
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
    async def tts_cooldown(
        self, event: AstrMessageEvent, *, value: Optional[str] = None
    ):
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

    @filter.command("tts_test", priority=1)
    async def tts_test(self, event: AstrMessageEvent, *, text: Optional[str] = None):
        """测试TTS功能并诊断问题。用法：tts_test [测试文本]"""
        if not text:
            text = "你好，这是一个TTS测试"
        
        sid = self._sess_id(event)
        if not self._is_session_enabled(sid):
            yield event.plain_result("本会话TTS未启用，请使用 tts_on 启用")
            return
        
        try:
            # 选择默认情绪和音色
            emotion = "neutral"
            vkey, voice = self._pick_voice_for_emotion(emotion)
            if not voice:
                yield event.plain_result(f"错误：未配置音色映射，请先配置 voice_map.{emotion}")
                return
            
            # 创建输出目录
            out_dir = TEMP_DIR / sid
            ensure_dir(out_dir)
            
            # 生成音频
            yield event.plain_result(f"正在生成测试音频：\"{text}\"...")
            
            start_time = time.time()
            audio_path = self.tts.synth(text, voice, out_dir, speed=None)
            generation_time = time.time() - start_time
            
            if not audio_path:
                yield event.plain_result("❌ TTS API调用失败")
                return
            
            # 验证文件
            if not self._validate_audio_file(audio_path):
                yield event.plain_result(f"❌ 生成的音频文件无效: {audio_path}")
                return
            
            # 路径规范化测试
            normalized_path = self._normalize_audio_path(audio_path)
            
            # 尝试创建Record对象
            try:
                record = Record(file=normalized_path)
                record_status = "✅ 成功"
            except Exception as e:
                record_status = f"❌ 失败: {e}"
            
            # 报告结果
            file_size = audio_path.stat().st_size
            result_msg = f"""🎵 TTS测试结果：
✅ 音频生成成功
📁 文件路径: {audio_path.name}
📊 文件大小: {file_size} 字节
⏱️ 生成耗时: {generation_time:.2f}秒
🎯 使用音色: {vkey} ({voice[:30]}...)
📝 Record对象: {record_status}
🔧 规范化路径: {normalized_path == str(audio_path)}"""
            
            yield event.plain_result(result_msg)
            
            # 尝试发送音频
            try:
                yield event.record_result(str(audio_path))
            except Exception as e:
                yield event.plain_result(f"❌ 音频发送失败: {e}")
            
        except Exception as e:
            yield event.plain_result(f"❌ TTS测试失败: {e}")
            logging.error(f"TTS测试异常: {e}", exc_info=True)

    @filter.command("tts_debug", priority=1)
    async def tts_debug(self, event: AstrMessageEvent):
        """显示TTS调试信息"""
        try:
            sid = self._sess_id(event)
            st = self._session_state.get(sid, SessionState())
            
            # 系统信息
            import platform
            import os
            
            debug_info = f"""🔧 TTS调试信息：
🖥️ 系统: {platform.system()} {platform.release()}
📂 Python路径: {os.getcwd()}
🆔 会话ID: {sid}
⚡ 会话状态: {'✅ 启用' if self._is_session_enabled(sid) else '❌ 禁用'}
🎛️ 全局开关: {'✅ 开启' if self.global_enable else '❌ 关闭'}
🎲 触发概率: {self.prob}
📏 文字限制: {self.text_limit}
⏰ 冷却时间: {self.cooldown}s
🔄 混合内容: {'✅ 允许' if self.allow_mixed else '❌ 禁止'}
🎵 API模型: {self.tts.model}
🎚️ 音量增益: {self.tts.gain}dB
📁 临时目录: {TEMP_DIR}

📊 会话统计:
🕐 最后TTS时间: {time.strftime('%H:%M:%S', time.localtime(st.last_tts_time)) if st.last_tts_time else '无'}
📝 最后TTS内容: {st.last_tts_content[:30] + '...' if st.last_tts_content and len(st.last_tts_content) > 30 else st.last_tts_content or '无'}
😊 待用情绪: {st.pending_emotion or '无'}

🎭 音色配置:"""
            
            for emotion in EMOTIONS:
                vkey, voice = self._pick_voice_for_emotion(emotion)
                speed = self.speed_map.get(emotion) if isinstance(self.speed_map, dict) else None
                debug_info += f"\n{emotion}: {vkey if voice else '❌ 未配置'}"
                if speed:
                    debug_info += f" (语速: {speed})"
            
            yield event.plain_result(debug_info)
            
        except Exception as e:
            yield event.plain_result(f"❌ 获取调试信息失败: {e}")

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
        yield event.plain_result(
            f"模式: {mode}\n当前会话: {'启用' if enabled else '禁用'}\nprob={self.prob}, limit={self.text_limit}, cooldown={self.cooldown}s, allow_mixed={self.allow_mixed}"
        )

    @filter.command("tts_mixed_on", priority=1)
    async def tts_mixed_on(self, event: AstrMessageEvent):
        """允许混合输出（文本+语音都保留）"""
        self.allow_mixed = True
        try:
            if self.config is not None and (
                isinstance(self.config, AstrBotConfig) or isinstance(self.config, dict)
            ):
                self.config["allow_mixed"] = True
                self._save_config()
        except Exception:
            pass
        yield event.plain_result("TTS混合输出：开启（文本+语音）")

    @filter.command("tts_mixed_off", priority=1)
    async def tts_mixed_off(self, event: AstrMessageEvent):
        """仅纯文本可参与合成；含图片/回复等时跳过"""
        self.allow_mixed = False
        try:
            if self.config is not None and (
                isinstance(self.config, AstrBotConfig) or isinstance(self.config, dict)
            ):
                self.config["allow_mixed"] = False
                self._save_config()
        except Exception:
            pass
        yield event.plain_result("TTS混合输出：关闭（仅纯文本时尝试合成）")


    @filter.command("tts_check_refs", priority=1)
    async def tts_check_refs(self, event: AstrMessageEvent):
        """检查参考文献配置"""
        yield event.plain_result(
            f"allow_mixed配置: {self.allow_mixed}\n"
            f"配置文件中的allow_mixed: {self.config.get('allow_mixed', '未找到')}\n"
            f"参考文献发送条件: {'满足' if self.allow_mixed else '不满足 (需要开启 allow_mixed)'}"
        )

    # ---------------- After send hook: 防止重复 RespondStage 再次发送 -----------------
    # 兼容不同 AstrBot 版本：优先使用 after_message_sent，其次回退 on_after_message_sent；都没有则不挂载该钩子。
    if hasattr(filter, "after_message_sent"):
        @filter.after_message_sent(priority=-1000)
        async def after_message_sent(self, event: AstrMessageEvent):
            # 仅记录诊断信息，不再清空链，避免影响历史写入/上下文。
            try:
                # 确保不被判定为终止传播
                try:
                    event.continue_event()
                except Exception:
                    pass
                try:
                    res = event.get_result()
                    # 只读，不创建/修改 result，避免触发重复发送
                    if res is not None and hasattr(res, "continue_event"):
                        res.continue_event()
                except Exception:
                    pass
                try:
                    logging.debug("TTSEmotionRouter.after_message_sent: entry is_stopped=%s", event.is_stopped())
                except Exception:
                    pass
                result = event.get_result()
                if not result or not getattr(result, "chain", None):
                    return
                try:
                    has_plain = any(isinstance(c, Plain) for c in result.chain)
                    has_record = any(isinstance(c, Record) for c in result.chain)
                    logging.info(
                        "after_message_sent: snapshot len=%d, has_plain=%s, has_record=%s, is_llm=%s",
                        len(result.chain), has_plain, has_record, getattr(result, "result_content_type", None) == ResultContentType.LLM_RESULT,
                    )
                except Exception:
                    pass
                # 兜底：若为 LLM 结果且包含任意语音（不局限于本插件），确保将可读文本写入对话历史
                try:
                    if any(isinstance(c, Record) for c in result.chain):
                        await self._ensure_history_saved(event)
                except Exception:
                    pass
                # 再次声明继续传播
                try:
                    event.continue_event()
                except Exception:
                    pass
                try:
                    res = event.get_result()
                    if res is not None and hasattr(res, "continue_event"):
                        res.continue_event()
                except Exception:
                    pass
                # 兼容部分框架对“未产出/未修改”的停止判定，进行一次无害的 get_result 访问
                try:
                    _ = event.get_result()
                except Exception:
                    pass
                try:
                    logging.debug("TTSEmotionRouter.after_message_sent: exit is_stopped=%s", event.is_stopped())
                except Exception:
                    pass
            except Exception:
                pass
    elif hasattr(filter, "on_after_message_sent"):
        @filter.on_after_message_sent(priority=-1000)
        async def after_message_sent(self, event: AstrMessageEvent):
            # 仅记录诊断信息，不再清空链，避免影响历史写入/上下文。
            try:
                # 确保不被判定为终止传播
                try:
                    event.continue_event()
                except Exception:
                    pass
                try:
                    res = event.get_result()
                    # 只读，不创建/修改 result，避免触发重复发送
                    if res is not None and hasattr(res, "continue_event"):
                        res.continue_event()
                except Exception:
                    pass
                try:
                    logging.debug("TTSEmotionRouter.after_message_sent: entry(is_compat) is_stopped=%s", event.is_stopped())
                except Exception:
                    pass
                result = event.get_result()
                if not result or not getattr(result, "chain", None):
                    return
                try:
                    has_plain = any(isinstance(c, Plain) for c in result.chain)
                    has_record = any(isinstance(c, Record) for c in result.chain)
                    logging.info(
                        "after_message_sent: snapshot len=%d, has_plain=%s, has_record=%s, is_llm=%s",
                        len(result.chain), has_plain, has_record, getattr(result, "result_content_type", None) == ResultContentType.LLM_RESULT,
                    )
                except Exception:
                    pass
                # 兜底：若为 LLM 结果且包含任意语音（不局限于本插件），确保将可读文本写入对话历史
                try:
                    if any(isinstance(c, Record) for c in result.chain):
                        await self._ensure_history_saved(event)
                except Exception:
                    pass
                # 再次声明继续传播
                try:
                    event.continue_event()
                except Exception:
                    pass
                try:
                    res = event.get_result()
                    if res is not None and hasattr(res, "continue_event"):
                        res.continue_event()
                except Exception:
                    pass
                # 兼容部分框架对“未产出/未修改”的停止判定，进行一次无害的 get_result 访问
                try:
                    _ = event.get_result()
                except Exception:
                    pass
                try:
                    logging.debug("TTSEmotionRouter.after_message_sent: exit(is_compat) is_stopped=%s", event.is_stopped())
                except Exception:
                    pass
            except Exception:
                pass
    else:
        async def after_message_sent(self, event: AstrMessageEvent):
            return

    # ---------------- Core hook -----------------
    @filter.on_decorating_result(priority=-1000)
    async def on_decorating_result(self, event: AstrMessageEvent):
        # 在入口处尽可能声明继续传播，避免被归因为终止传播
        try:
            event.continue_event()
        except Exception:
            pass
        try:
            logging.info("TTSEmotionRouter.on_decorating_result: entry is_stopped=%s", event.is_stopped())
        except Exception:
            pass
        # 若进入本阶段已为 STOP，主动切回 CONTINUE
        try:
            if event.is_stopped():
                logging.info("TTSEmotionRouter.on_decorating_result: detected STOP at entry, forcing CONTINUE for decorating")
                event.continue_event()
        except Exception:
            pass

        sid = self._sess_id(event)
        if not self._is_session_enabled(sid):
            logging.info("TTS skip: session disabled (%s)", sid)
            try:
                event.continue_event()
            except Exception:
                pass
            return

        # 结果链
        result = event.get_result()
        if not result or not result.chain:
            logging.debug("TTS skip: empty result chain")
            try:
                event.continue_event()
            except Exception:
                pass
            return

        # 清理首个 Plain 的隐藏情绪头
        try:
            new_chain = []
            cleaned_once = False
            for comp in result.chain:
                if (
                    not cleaned_once
                    and isinstance(comp, Plain)
                    and getattr(comp, "text", None)
                ):
                    t0 = self._normalize_text(comp.text)
                    t, _ = self._strip_emo_head_many(t0)
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
            logging.info("TTS skip: mixed content not allowed (allow_mixed=%s)", self.allow_mixed)
            try:
                event.continue_event()
            except Exception:
                pass
            return

        # 拼接纯文本
        text_parts = [
            c.text.strip()
            for c in result.chain
            if isinstance(c, Plain) and c.text.strip()
        ]
        if not text_parts:
            logging.debug("TTS skip: no plain text parts after cleaning")
            try:
                event.continue_event()
            except Exception:
                pass
            return
        text = " ".join(text_parts)

        # 归一化 + 连续剥离（终极兜底）
        orig_text = text
        text = self._normalize_text(text)
        text, _ = self._strip_emo_head_many(text)

        # 提取代码和链接
        extracted_content = extractor.extract_all(text)
        logging.info(f"TTS: extracted_content count={len(extracted_content)}, text={text[:50]}...")
        
        # 如果有代码或链接，准备参考文献
        references = ""
        if extracted_content:
            references = extractor.format_references(extracted_content)
            logging.info(f"TTS: generated references length={len(references)}")
            logging.info(f"TTS: references content preview={references[:200]}...")
        
        # 清理文本用于TTS
        tts_text = extractor.clean_text_for_tts(text)
        logging.info(f"TTS: tts_text length={len(tts_text)}, content={tts_text[:50]}...")
        
        # 检查是否还有文本需要朗读
        if not tts_text.strip():
            logging.info("TTS skip: no text left after cleaning")
            try:
                event.continue_event()
            except Exception:
                pass
            return
        
        # 注意：不再跳过TTS生成，而是继续执行后续流程
        # 代码和链接的处理将在生成音频后，在结果链中添加参考文献

        # 去重逻辑已移除：总是继续尝试合成
        st = self._session_state.setdefault(sid, SessionState())
        now = time.time()
        if self.cooldown > 0 and (now - st.last_ts) < self.cooldown:
            logging.info("TTS skip: cooldown active (%.2fs < %ss)", now - st.last_ts, self.cooldown)
            try:
                event.continue_event()
            except Exception:
                pass
            return

        if self.text_limit > 0 and len(tts_text) > self.text_limit:
            logging.info("TTS skip: over text_limit (len=%d > limit=%d)", len(tts_text), self.text_limit)
            try:
                event.continue_event()
            except Exception:
                pass
            return

        if random.random() > self.prob:
            logging.info("TTS skip: probability gate (prob=%.2f)", self.prob)
            try:
                event.continue_event()
            except Exception:
                pass
            return

        # 情绪选择：优先使用隐藏标记 -> 启发式
        if st.pending_emotion in EMOTIONS:
            emotion = st.pending_emotion
            st.pending_emotion = None
            src = "tag"
        else:
            emotion = self.heuristic_cls.classify(tts_text, context=None)
            src = "heuristic"
            try:
                kw = getattr(self, "_emo_kw", {})
                has_kw = any(p.search(tts_text) for p in kw.values())
                if not has_kw:
                    emotion = "neutral"
            except Exception:
                pass

        vkey, voice = self._pick_voice_for_emotion(emotion)
        if not voice:
            logging.warning("No voice mapped for emotion=%s", emotion)
            try:
                event.continue_event()
            except Exception:
                pass
            return

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

        logging.info(
            "TTS route: emotion=%s(src=%s) -> %s (%s), speed=%s",
            emotion,
            src,
            vkey,
            (voice[:40] + "...") if isinstance(voice, str) and len(voice) > 43 else voice,
            speed_override if speed_override is not None else getattr(self.tts, "speed", None),
        )
        logging.debug("TTS input head(before/after): %r -> %r", orig_text[:60], tts_text[:60])

        out_dir = TEMP_DIR / sid
        ensure_dir(out_dir)

        # 最后一重防线：若 TTS 前文本仍以 emo/token 开头，强制清理
        try:
            if text and (text.lower().lstrip().startswith("emo") or text.lstrip().startswith(("[", "【", "("))):
                text, _ = self._strip_emo_head_many(text)
        except Exception:
            pass

        # 不做生成级去重：重复发送问题通过结果链策略规避

        try:
            audio_path = self.tts.synth(tts_text, voice, out_dir, speed=speed_override)
            if not audio_path:
                logging.error("TTS调用失败，降级为文本")
                # TTS失败时，如果有代码或链接，添加参考文献
                if references:
                    # 使用清理后的文本 + 参考文献
                    clean_text_for_display = extractor.clean_text_for_tts(orig_text)
                    # 如果清理后的文本为空，使用原始文本
                    if not clean_text_for_display.strip():
                        clean_text_for_display = orig_text
                    result.chain = [Plain(text=clean_text_for_display + references)]
                    logging.info(f"TTS失败: 输出文本+参考文献，文本长度={len(clean_text_for_display)}")
                else:
                    result.chain = [Plain(text=orig_text)]
                    logging.info("TTS失败: 输出纯文本")
                try:
                    event.continue_event()
                except Exception:
                    pass
                return

            # === 专门针对retcode=1200问题的增强处理 ===
            
            # 1. 验证生成的音频文件
            if not self._validate_audio_file(audio_path):
                logging.error(f"TTS生成的音频文件无效: {audio_path}")
                # 直接回退到文本，不发送无效音频
                result.chain = [Plain(text=text)]
                try:
                    event.continue_event()
                except Exception:
                    pass
                return
            
            # 2. 使用相对路径以提高兼容性
            try:
                # 计算相对于工作目录的路径
                import os
                work_dir = Path(os.getcwd())
                try:
                    relative_path = audio_path.relative_to(work_dir)
                    audio_file_path = str(relative_path).replace('\\', '/')
                    logging.info(f"TTS: 使用相对路径: {audio_file_path}")
                except ValueError:
                    # 如果无法计算相对路径，使用绝对路径
                    audio_file_path = str(audio_path).replace('\\', '/')
                    logging.info(f"TTS: 使用绝对路径: {audio_file_path}")
            except Exception:
                audio_file_path = str(audio_path)
            
            # 3. 创建Record对象前进行最后验证
            try:
                # 确保文件存在且可读
                test_path = Path(audio_file_path) if not Path(audio_file_path).is_absolute() else audio_path
                if not test_path.exists():
                    raise FileNotFoundError(f"音频文件不存在: {test_path}")
                
                # 检查文件大小
                file_size = test_path.stat().st_size
                if file_size == 0:
                    raise ValueError(f"音频文件为空: {test_path}")
                
                logging.info(f"TTS: 音频文件验证通过，大小={file_size}字节")
                
            except Exception as e:
                logging.error(f"TTS: 音频文件验证失败: {e}")
                # 验证失败时回退到纯文本
                result.chain = [Plain(text=tts_text)]
                try:
                    event.continue_event()
                except Exception:
                    pass
                return
            
            # 4. 使用更保守的Record创建策略
            try:
                record = Record(file=audio_file_path)
                logging.info(f"TTS: 成功创建Record对象，路径={audio_file_path}")
                
                # 更新会话状态
                st.last_tts_content = tts_text
                st.last_tts_time = time.time()
                st.last_ts = time.time()

                # 如果有代码或链接，只发送参考文献（不包含原文）
                if references:
                    # TTS已经输出了完整内容，文本只补充参考文献
                    if references.strip():
                        logging.info("TTS: sending references only (no original text)")
                        result.chain = [Plain(text=references), record]
                    else:
                        logging.warning("TTS: references is empty, sending audio only")
                        result.chain = [record]
                else:
                    # 没有参考文献，根据 allow_mixed 决定是否同时发送文本
                    if self.allow_mixed:
                        result.chain = [Plain(text=orig_text), record]
                        logging.info("TTS: 输出混合内容，无参考文献")
                    else:
                        result.chain = [record]
                        logging.info("TTS: 输出纯音频")
                
                # 记录成功信息
                logging.info(f"TTS: 音频处理完成 - 文件={audio_path.name}, 大小={file_size}字节")
                
            except Exception as e:
                logging.error(f"TTS: 创建Record失败: {e}")
                # Record创建失败，强制回退到文本
                result.chain = [Plain(text=tts_text)]
                logging.info("TTS: 已回退到纯文本输出")

            # 5. 统一的后续处理
            try:
                _hp = any(isinstance(c, Plain) for c in result.chain)
                _hr = any(isinstance(c, Record) for c in result.chain)
                logging.info("TTS finalize: has_plain=%s, has_record=%s, text_len=%d", _hp, _hr, len(text))
            except Exception:
                pass

            try:
                _ = await self._append_assistant_text_to_history(event, text)
            except Exception:
                pass
            try:
                event.continue_event()
            except Exception:
                pass
            try:
                st.last_assistant_text = text.strip()
                st.last_assistant_text_time = time.time()
            except Exception:
                pass
            try:
                result.set_result_content_type(ResultContentType.LLM_RESULT)
            except Exception:
                pass
            # 明确声明结果未停止
            try:
                event.continue_event()
            except Exception:
                pass
            return
        finally:
            try:
                event.continue_event()
            except Exception:
                pass
        try:
            logging.info("TTSEmotionRouter.on_decorating_result: exit is_stopped=%s", event.is_stopped())
            event.continue_event()
        except Exception:
            pass

    async def _ensure_history_saved(self, event: AstrMessageEvent) -> None:
        """兜底：保证本轮助手可读文本写入到会话历史。
        条件：当前结果被标记为 LLM_RESULT，且链中含有本插件生成的 Record。
        逻辑：聚合链中的 Plain 文本；若历史最后的 assistant 文本不等于该文本，则补记一条。
        """
        try:
            result = event.get_result()
            if not result or not getattr(result, "chain", None):
                return
            # 兼容不同 AstrBot 版本：若无法判断 is_llm_result，则仅以“链中含本插件音频”为条件。
            is_llm = False
            try:
                is_llm = bool(result.is_llm_result())
            except Exception:
                is_llm = False
            if not is_llm and not any(self._is_our_record(c) for c in result.chain):
                return
            # 聚合文本
            parts = []
            for comp in result.chain:
                if isinstance(comp, Plain) and getattr(comp, "text", None):
                    t = comp.text.strip()
                    if t:
                        parts.append(t)
            text = "\n".join(parts).strip()
            if not text:
                # 若链中没有文本，回退使用缓存
                try:
                    sid = self._sess_id(event)
                    st = self._session_state.setdefault(sid, SessionState())
                    if st.last_assistant_text and (time.time() - st.last_assistant_text_time) < 60:
                        await self._append_assistant_text_to_history(event, st.last_assistant_text)
                except Exception:
                    pass
                return
            await self._append_assistant_text_to_history(event, text)
        except Exception:
            # 容错：不因兜底写入失败影响主流程
            pass

    async def _append_assistant_text_to_history(self, event: AstrMessageEvent, text: str) -> bool:
        """使用已清洗的最终文本，直接写入会话历史（去重且幂等）。返回是否成功写入。"""
        if not text:
            return False
        try:
            cm = self.context.conversation_manager
            uid = event.unified_msg_origin
            # 获取会话ID：优先 provider_request，其次当前活跃会话；若暂不可用，小退避重试
            cid = None
            for attempt in range(3):
                try:
                    req = getattr(event, "get_extra", None) and event.get_extra("provider_request")
                    if req and getattr(req, "conversation", None) and getattr(req.conversation, "cid", None):
                        cid = req.conversation.cid
                except Exception:
                    cid = None
                if not cid:
                    try:
                        cid = await cm.get_curr_conversation_id(uid)
                    except Exception:
                        cid = None
                if cid:
                    break
                # 等待会话在核心落库
                await asyncio.sleep(0.2)
            if not cid:
                logging.info("TTSEmotionRouter.history_fallback: skip write, no active conversation id after retry")
                return False
            # 获取会话体，优先不创建；若仍未就绪，小退避后允许创建一次，避免错过本轮文本
            conv = await cm.get_conversation(uid, cid, create_if_not_exists=False)
            if not conv:
                await asyncio.sleep(0.2)
                try:
                    conv = await cm.get_conversation(uid, cid, create_if_not_exists=True)
                except Exception:
                    conv = None
            if not conv:
                logging.info("TTSEmotionRouter.history_fallback: conversation still not available for cid=%s", cid)
                return False
            import json as _json
            msgs = []
            try:
                msgs = _json.loads(conv.history) if getattr(conv, "history", "") else []
            except Exception:
                msgs = []

            # 若最后一个 assistant 文本已相同，则不重复写入
            if msgs:
                last = msgs[-1]
                if isinstance(last, dict) and last.get("role") == "assistant" and (last.get("content") or "").strip() == text.strip():
                    return True

            msgs.append({"role": "assistant", "content": text.strip()})
            await cm.update_conversation(uid, cid, history=msgs)
            logging.info("TTSEmotionRouter.history_fallback: appended assistant text to conversation history")
            return True
        except Exception:
            return False

    async def _delayed_history_write(self, event: AstrMessageEvent, text: str, delay: float = 0.8):
        """延迟写入一次会话历史，避免 on_llm_response 时会话尚未建立导致的落库失败。"""
        try:
            await asyncio.sleep(max(0.0, float(delay)))
            await self._append_assistant_text_to_history(event, text)
        except Exception:
            pass
