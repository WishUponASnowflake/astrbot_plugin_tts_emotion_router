# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ExtractedContent:
    """提取的代码或链接内容"""
    type: str  # "code" 或 "link"
    content: str
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    language: Optional[str] = None


class CodeAndLinkExtractor:
    """提取文本中的代码块和链接"""
    
    def __init__(self):
        # 代码块正则 - 匹配 ```language\n...\n``` 或 `inline code`
        self.code_block_re = re.compile(
            r'```([a-zA-Z0-9_+-]*)\n(.*?)\n```',
            re.DOTALL
        )
        self.inline_code_re = re.compile(
            r'`([^`\n]+)`'
        )
        
        # 链接正则
        self.url_re = re.compile(
            r'https?://[^\s<>"{}|\\^`\[\]]+',
            re.IGNORECASE
        )
        self.www_re = re.compile(
            r'www\.[^\s<>"{}|\\^`\[\]]+\.[^\s<>"{}|\\^`\[\]]+',
            re.IGNORECASE
        )
        # 网址正则（匹配 xxx.xxx 格式）
        self.website_re = re.compile(
            r'[a-zA-Z0-9][a-zA-Z0-9-]{1,61}[a-zA-Z0-9]\.[a-zA-Z]{2,}(?:\.[a-zA-Z]{2,})?',
            re.IGNORECASE
        )
        
        # 常见的技术术语和模型名 - 这些不应该被视为代码
        self.tech_terms = {
            # AI模型
            'gpt', 'gpt-3.5', 'gpt-4', 'gpt-4-turbo', 'chatgpt',
            'claude', 'claude-2', 'claude-3', 'gemini', 'llama',
            'mistral', 'mixtral', 'qwen', 'baichuan', 'yi',
            'deepseek', 'codegeex', 'pangu', 'xunfei',
            # 常见技术名词
            'api', 'rest', 'json', 'xml', 'html', 'css', 'js',
            'python', 'java', 'javascript', 'typescript', 'c++',
            'tensorflow', 'pytorch', 'torch', 'numpy', 'pandas',
            'docker', 'kubernetes', 'k8s', 'linux', 'windows',
            'macos', 'ios', 'android', 'web', 'frontend', 'backend',
            # 版本号
            r'v?\d+\.\d+(\.\d+)?',
            r'\d+\.\d+(\.\d+)?',
        }
        
        # 编译技术术语正则
        self.tech_terms_re = re.compile(
            r'\b(' + '|'.join(re.escape(term) for term in self.tech_terms) + r')\b',
            re.IGNORECASE
        )
    
    def extract_code_blocks(self, text: str) -> List[ExtractedContent]:
        """提取代码块"""
        results = []
        
        # 提取多行代码块
        for match in self.code_block_re.finditer(text):
            language = match.group(1).strip() or None
            code_content = match.group(2)
            
            # 如果代码内容看起来像是技术术语而非实际代码，跳过
            if self._is_likely_tech_term(code_content):
                continue
                
            results.append(ExtractedContent(
                type="code",
                content=code_content,
                language=language
            ))
        
        # 提取行内代码
        for match in self.inline_code_re.finditer(text):
            code_content = match.group(1)
            
            # 如果是单个技术术语，跳过
            if self._is_likely_tech_term(code_content):
                continue
                
            # 如果是版本号或简单的技术引用，跳过
            if self._is_simple_tech_reference(code_content):
                continue
                
            results.append(ExtractedContent(
                type="code",
                content=code_content
            ))
        
        return results
    
    def extract_links(self, text: str) -> List[ExtractedContent]:
        """提取链接"""
        results = []
        
        # 提取完整URL
        for match in self.url_re.finditer(text):
            results.append(ExtractedContent(
                type="link",
                content=match.group(0)
            ))
        
        # 提取www开头的链接
        for match in self.www_re.finditer(text):
            results.append(ExtractedContent(
                type="link",
                content=match.group(0)
            ))
        
        # 提取网址格式（如 github.com）
        for match in self.website_re.finditer(text):
            # 排除一些常见的技术术语
            domain = match.group(0).lower()
            if domain not in ['api.com', 'example.com', 'test.com'] and not domain.endswith('.js'):
                results.append(ExtractedContent(
                    type="link",
                    content=match.group(0)
                ))
        
        return results
    
    def extract_all(self, text: str) -> List[ExtractedContent]:
        """提取所有代码和链接"""
        code_blocks = self.extract_code_blocks(text)
        links = self.extract_links(text)
        return code_blocks + links
    
    def clean_text_for_tts(self, text: str) -> str:
        """清理文本，移除代码块和链接，准备用于TTS"""
        # 移除多行代码块
        text = self.code_block_re.sub('', text)
        
        # 移除行内代码（保留技术术语）
        text = self.inline_code_re.sub(
            lambda m: m.group(1) if self._is_likely_tech_term(m.group(1)) else '',
            text
        )
        
        # 移除链接
        text = self.url_re.sub('', text)
        text = self.www_re.sub('', text)
        text = self.website_re.sub('', text)
        
        # 清理多余的空白
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def format_references(self, extracts: List[ExtractedContent]) -> str:
        """将提取的内容格式化为参考文献"""
        if not extracts:
            return ""
        
        references = []
        code_count = 0
        link_count = 0
        
        for extract in extracts:
            if extract.type == "code":
                code_count += 1
                if extract.language:
                    references.append(f"[代码{code_count}] {extract.language}代码片段")
                else:
                    references.append(f"[代码{code_count}] 代码片段")
                # 添加代码内容（截断过长的代码）
                code_preview = extract.content.replace('\n', ' ')
                if len(code_preview) > 100:
                    code_preview = code_preview[:97] + "..."
                references[-1] += f": {code_preview}"
                    
            elif extract.type == "link":
                link_count += 1
                references.append(f"[链接{link_count}] {extract.content}")
        
        if references:
            return "\n\n📚 参考文献和代码片段：\n" + "\n".join(references)
        return ""
    
    def _is_likely_tech_term(self, text: str) -> bool:
        """判断是否为技术术语而非实际代码"""
        text = text.strip().lower()
        
        # 如果是单个词且匹配技术术语
        if ' ' not in text and self.tech_terms_re.fullmatch(text):
            return True
            
        # 如果是简单的版本号或模型名
        if re.match(r'^[a-zA-Z-]+/\d+\.\d+$', text):  # 如 openai/3.5
            return True
            
        return False
    
    def _is_simple_tech_reference(self, text: str) -> bool:
        """判断是否为简单的技术引用"""
        text = text.strip()
        
        # 如果只是模型名或版本号
        if re.match(r'^(gpt|claude|gemini|llama|qwen|yi|deepseek)-?\d*\.?\d*$', text, re.I):
            return True
            
        # 如果只是简单的版本号
        if re.match(r'^v?\d+\.\d+(\.\d+)?$', text):
            return True
            
        return False


# 全局实例
extractor = CodeAndLinkExtractor()