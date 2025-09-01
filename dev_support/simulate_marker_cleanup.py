#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""快速模拟：测试情绪标记清理行为 (已移至 dev_support)。"""
from pathlib import Path
import sys, importlib

CURR = Path(__file__).parent.resolve()
PLUGIN_ROOT = CURR.parent  # 插件根目录
REPO_ROOT = PLUGIN_ROOT.parents[3]  # 近似定位到 AstrBot-Complete 根
for p in {PLUGIN_ROOT, REPO_ROOT}:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

mod = importlib.import_module('data.plugins.astrbot_plugin_tts_emotion_router.main')
TTSEmotionRouter = getattr(mod, 'TTSEmotionRouter')
Context = getattr(mod, 'Context')


class DummyContext(Context):  # 仅提供最少接口
    def __init__(self):
        pass


if __name__ == '__main__':
    ctx = DummyContext()
    star = TTSEmotionRouter(ctx, config={})
    samples = [
        '[EMO:happy] 今天天气真好，我们去公园吧！',
        '【EMO：sad】唉，今天有点累。[EMO:happy]尾部又插入',
        '(EMO:angry)我真的要生气了！',
        '[EMO:neutral][EMO:happy] 叠加标记处理后应该只剩正文',
        '正常句子，不含标记',
    ]
    for s in samples:
        cleaned, label = star._strip_emo_head_many(s)
        final = star._strip_any_visible_markers(cleaned)
        print('\n原文 :', s)
        print('初步清理:', cleaned, 'label=', label)
        print('最终清理:', final)
