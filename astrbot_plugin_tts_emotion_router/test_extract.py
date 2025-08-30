#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试代码和链接提取功能"""

import sys
from pathlib import Path

# 添加插件路径到Python路径
plugin_dir = Path(__file__).parent
sys.path.insert(0, str(plugin_dir))

from utils.extract import extractor

def test_extract():
    print("=== 测试代码和链接提取功能 ===\n")
    
    # 测试用例
    test_cases = [
        {
            "name": "包含模型名但不应该被识别为代码",
            "text": "我推荐使用 GPT-4 模型，它的效果很好。Claude-3 也不错。",
            "should_extract": False
        },
        {
            "name": "包含真正的代码块",
            "text": "这是一个Python代码示例：\n```python\ndef hello():\n    print(\"Hello, World!\")\n```\n这段代码会打印问候语。",
            "should_extract": True
        },
        {
            "name": "包含行内代码",
            "text": "使用 `print()` 函数可以输出内容，而 `input()` 可以获取用户输入。",
            "should_extract": True
        },
        {
            "name": "包含链接",
            "text": "更多信息请访问 https://example.com 或 www.example.org",
            "should_extract": True
        },
        {
            "name": "混合内容",
            "text": "我推荐使用 GPT-4 模型。示例代码：\n```python\nimport openai\nresponse = openai.ChatCompletion.create(\n    model=\"gpt-4\",\n    messages=[{\"role\": \"user\", \"content\": \"Hello\"}]\n)\n```\n详情见：https://platform.openai.com/docs",
            "should_extract": True
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"测试用例 {i}: {case['name']}")
        print(f"输入文本: {case['text'][:50]}...")
        
        # 提取内容
        extracted = extractor.extract_all(case['text'])
        print(f"提取结果: {len(extracted)} 项")
        for item in extracted:
            print(f"  - {item.type}: {item.content[:50]}...")
        
        # 清理文本
        cleaned = extractor.clean_text_for_tts(case['text'])
        print(f"清理后文本: {cleaned[:50]}...")
        
        # 生成参考文献
        refs = extractor.format_references(extracted)
        if refs:
            print(f"参考文献: {refs[:100]}...")
        
        # 验证结果
        if case['should_extract']:
            assert len(extracted) > 0, f"应该提取到内容但没有提取到"
            print("[PASS] 测试通过")
        else:
            # 模型名等应该被保留，不应该被提取
            assert len(extracted) == 0, f"不应该提取到内容但提取了 {len(extracted)} 项"
            # 确保模型名还在清理后的文本中
            assert "GPT-4" in cleaned, "模型名应该被保留"
            print("[PASS] 测试通过")
        
        print("-" * 60)

if __name__ == "__main__":
    test_extract()