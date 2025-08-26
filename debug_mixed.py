#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试混合格式问题
"""
import re

def debug_mixed_format():
    text = "【情绪：开心】&shy& 全部清理"
    print(f"原始: '{text}'")
    
    patterns = [
        r'^\s*\[?\s*emo\s*[:：]?\s*\w*\s*\]?\s*[,，。:\uff1a]*\s*',
        r'^\s*\[?\s*EMO\s*[:：]?\s*\w*\s*\]?\s*[,，。:\uff1a]*\s*',
        r'^\s*【\s*[Ee][Mm][Oo]\s*[:：]?\s*\w*\s*】\s*[,，。:\uff1a]*\s*',
        r'\[情绪[:：]\w*\]',
        r'\[心情[:：]\w*\]',
        r'^\s*情绪[:：]\s*\w+\s*[,，。]\s*',
        r'&[a-zA-Z\u4e00-\u9fff]+&',
        r'^\s*&[a-zA-Z\u4e00-\u9fff]+&\s*[,，。:\uff1a]*\s*',
    ]
    
    current_text = text
    for i, pattern in enumerate(patterns):
        before = current_text
        current_text = re.sub(pattern, '', current_text, flags=re.IGNORECASE)
        if before != current_text:
            print(f"模式{i+1} ({pattern[:20]}...): '{before}' -> '{current_text}'")
    
    # 测试专门的【情绪：开心】模式
    special_pattern = r'【情绪[:：][^】]*】'
    print(f"\n测试专门模式: {special_pattern}")
    result = re.sub(special_pattern, '', text)
    print(f"结果: '{text}' -> '{result}'")

if __name__ == "__main__":
    debug_mixed_format()