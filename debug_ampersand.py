#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试&标签过滤功能
"""
import re

def debug_ampersand_filtering():
    """调试&标签过滤"""
    
    # 测试用例
    test_cases = [
        "前面文字 &sad& 后面文字",
        "开始 &joy& 中间 &worry& 结束", 
        "text &confused& more text",
        "【情绪：开心】&shy& 全部清理",
    ]
    
    # 当前的正则模式
    patterns = [
        r'^\s*\[?\s*emo\s*[:：]?\s*\w*\s*\]?\s*[,，。:\uff1a]*\s*',
        r'^\s*\[?\s*EMO\s*[:：]?\s*\w*\s*\]?\s*[,，。:\uff1a]*\s*',
        r'^\s*【\s*[Ee][Mm][Oo]\s*[:：]?\s*\w*\s*】\s*[,，。:\uff1a]*\s*',
        r'\[情绪[:：]\w*\]',
        r'\[心情[:：]\w*\]',
        r'^\s*情绪[:：]\s*\w+\s*[,，。]\s*',
        r'&[a-zA-Z\u4e00-\u9fff]+&',  # &英文或中文&
        r'^\s*&[a-zA-Z\u4e00-\u9fff]+&\s*[,，。:\uff1a]*\s*',  # 开头的&标签&
    ]
    
    for text in test_cases:
        print(f"\n原始: '{text}'")
        
        current_text = text
        for i, pattern in enumerate(patterns):
            before = current_text
            current_text = re.sub(pattern, '', current_text, flags=re.IGNORECASE)
            if before != current_text:
                print(f"模式{i+1}: '{before}' -> '{current_text}'")
        
        # 清理多余空格
        cleaned = re.sub(r'\s+', ' ', current_text).strip()
        print(f"清理空格后: '{cleaned}'")
        
        # 添加结尾
        if cleaned and not re.search(r'[。！？.!?，,]$', cleaned):
            if re.search(r'[\u4e00-\u9fff]', cleaned):
                cleaned += '。'
            else:
                cleaned += '.'
        if cleaned and not cleaned.endswith('...'):
            cleaned += '..'
        
        print(f"最终结果: '{cleaned}'")

if __name__ == "__main__":
    debug_ampersand_filtering()