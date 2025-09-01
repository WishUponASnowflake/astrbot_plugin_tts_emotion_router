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

        # 允许的常见顶级域（用于过滤伪匹配如 plugin.html）
        self.allowed_tlds = {
            'com','net','org','io','app','dev','cn','ai','co','gov','edu','uk','de','jp','us','info','xyz','top','site','club'
        }
    
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

            # 如果不像真正的代码（只是普通名称/词语），跳过
            if not self._looks_like_real_code(code_content):
                continue

            # 过滤琐碎片段（纯扩展名、简单文件名等）
            if self._is_trivial_inline(code_content):
                continue
                
            results.append(ExtractedContent(
                type="code",
                content=code_content
            ))
        
        return results
    
    def extract_links(self, text: str) -> List[ExtractedContent]:
        """提取链接"""
        results = []
        seen: set[str] = set()

        def _add(raw: str):
            url = self._normalize_link(raw)
            if not url:
                return
            # 过滤明显的伪域名（含大写/括号/下划线/看似方法调用的驼峰段）
            if self._is_false_positive_url(url):
                return
            # 过滤不在允许 TLD 列表中的纯域名/疑似域名 (plugin.html)
            if '://' not in url and '/' not in url:
                parts = url.lower().split('.')
                if len(parts) >= 2:
                    tld = parts[-1]
                    if tld not in self.allowed_tlds:
                        return
            # 若已有完整 URL，忽略其裸域名（避免重复 docs.a.com + https://docs.a.com/x）
            if '://' not in url and '/' not in url:
                # 是裸域，检查是否已存在更长 URL
                for existing in seen:
                    if existing.startswith(('http://','https://')) and existing.replace('http://','').replace('https://','').startswith(url + '/'):
                        return
            if url not in seen:
                seen.add(url)
                results.append(ExtractedContent(type="link", content=url))

        # 提取完整URL
        for match in self.url_re.finditer(text):
            _add(match.group(0))

        # 提取www开头的链接
        for match in self.www_re.finditer(text):
            _add(match.group(0))

        # 提取网址格式（如 github.com）
        for match in self.website_re.finditer(text):
            domain = match.group(0)
            low = domain.lower()
            if low in ['api.com', 'example.com', 'test.com']:
                continue
            if low.endswith('.js'):
                continue
            _add(domain)

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
    
    def format_references(self, extracts: List[ExtractedContent], preview_limit: int | None = None, max_total_chars: int | None = None) -> str:
        """将提取的内容格式化为参考文献
        :param preview_limit: 单段代码截断长度，0 或 None 表示不截断
        :param max_total_chars: 整个参考文献拼接后最大字符数，0 或 None 表示不限制
        """
        if not extracts:
            return ""
        
        references = []
        code_count = 0
        link_count = 0

        # 允许外部未指定时，通过全局配置（若主程序在调用时注入）
        if preview_limit is None:
            try:
                from ..main import TTSEmotionRouter  # type: ignore  # 避免循环风险：只在运行时
                # 尝试访问任意一个实例配置（这里无法直接取实例，保持 None 即不截断）
            except Exception:
                pass
        if max_total_chars is None:
            pass  # 同上，目前保持默认
        
        for extract in extracts:
            if extract.type == "code":
                code_count += 1
                if extract.language:
                    references.append(f"[代码{code_count}] {extract.language}代码片段")
                else:
                    references.append(f"[代码{code_count}] 代码片段")
                code_preview = extract.content.replace('\n', ' ')
                if preview_limit and preview_limit > 0 and len(code_preview) > preview_limit:
                    code_preview = code_preview[:preview_limit] + "..."
                references[-1] += f": {code_preview}"
                    
            elif extract.type == "link":
                link_count += 1
                references.append(f"[链接{link_count}] {extract.content}")
        
        result = "\n" + "\n".join(references) if references else ""

        if result and max_total_chars and max_total_chars > 0 and len(result) > max_total_chars:
            # 简单截断（保留开头），并在末尾加省略提示
            result = result[: max_total_chars - 3] + "..."

        return result
    
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

    def _looks_like_real_code(self, text: str) -> bool:
        """启发式判断行内反引号内容是否像真实代码/标识符
        目标：减少 `tavily`、`API Key` 这类普通词语被当成代码。
        条件（任意命中即认为是代码，否则视为普通文本）：
        1. 含有典型代码符号 (){}[]=.:;`'"<> 或 关键字片段 def/class/import/return/if/for/while
        2. 是带下划线/驼峰/数字的单个标识符（如 my_var, getUserName, TAVILY_API_KEY）且长度 >= 5
        3. 含有括号结尾的调用形式 foo() / print()
        4. 含有运算符 = + - * / % 或 -> =>
        被排除：
        - 仅由中文/空格/普通英文单词组成（1-3个词），且每个词长度 < 15
        - 全部为大写但无下划线且长度 < 5
        - 单个纯字母小写单词长度 < 5
        """
        t = text.strip()
        if not t:
            return False

        # 包含中文则认为不是代码
        if re.search(r'[\u4e00-\u9fff]', t):
            return False

        # 典型代码符号/关键字
        if re.search(r'[(){}\[\]=\.:;<>]|`', t):
            return True
        if re.search(r'\b(def|class|import|return|if|for|while|function|lambda)\b', t):
            return True
        if '->' in t or '=>' in t:
            return True
        if re.search(r'[=+\-*/%]', t):  # 赋值或算术
            return True

        # 多词情况：若是 1~3 个普通英文词（可能含大小写），判为非代码
        words = re.split(r'\s+', t)
        if 1 <= len(words) <= 3 and all(re.fullmatch(r'[A-Za-z]{1,15}', w) for w in words):
            # 例如 "API Key", "tavily", "Hello World"
            return False

        # 单个 token 情况
        if re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{1,50}', t):
            # 纯小写且短 -> 非代码
            if t.islower() and len(t) < 5:
                return False
            # 全大写且无下划线且短 -> 非代码
            if t.isupper() and '_' not in t and len(t) < 5:
                return False
            # 包含下划线/数字/驼峰混合 -> 视为代码/变量
            if '_' in t or any(ch.isdigit() for ch in t):
                return True
            if re.search(r'[a-z][A-Z]', t):  # 驼峰
                return True
            # 长度较长（>=10）也可能是类/函数名
            if len(t) >= 10:
                return True
            # 其余情况当作普通词
            return False

        # 默认不认为是代码
        return False

    def _is_trivial_inline(self, text: str) -> bool:
        """判断行内反引号内容是否过于琐碎不值得提取。
        排除：
        - 纯文件扩展名: .py / .js / .md 等
        - 简单文件名[字母数字下划线-]+.(py|js|ts|md|txt|json|yml|yaml) 且长度较短
        - 仅1~2字符的标记
        """
        t = text.strip()
        if len(t) <= 2:
            return True
        if re.fullmatch(r"\.[a-zA-Z0-9]{1,6}", t):
            return True
        if re.fullmatch(r"[A-Za-z0-9_-]{1,40}\.(py|js|ts|md|txt|json|yml|yaml)", t, re.I):
            return True
        return False

    # ---------------- 链接辅助 ----------------
    def _normalize_link(self, url: str) -> str:
        """去除链接末尾常见标点并统一大小写策略（保留原大小写但供后续过滤用）。"""
        url = url.strip()
        # 去除尾随标点 (.,)，但不处理 query/hash 内的
        url = re.sub(r'[\.,;:!?)+\]]+$', '', url)
        url = url.strip()
        return url

    def _is_false_positive_url(self, url: str) -> bool:
        """过滤非真实链接的伪匹配：
        1. 含空格直接判伪
        2. 不含协议且含大写字母（域名通常全小写；用户敲大写域一般仍可用，但为减少 openai.ChatCompletion 误判）
        3. 含括号/下划线/引号
        4. 形似方法调用：label.labelCamelCase 或 label.label_snake_case (无协议)
        5. 三个以上点分段且中间段含大写 -> 可能是模块/类链
        """
        if ' ' in url:
            return True
        # 拆掉协议
        no_proto = re.sub(r'^https?://', '', url, flags=re.I)
        # 方法调用式误判： openai.ChatCompletion.create
        if '://' not in url and re.search(r'[A-Z]', no_proto):
            # 允许全大写 TLD 的特殊情况（基本罕见，忽略）
            return True
        # 包含括号/下划线/引号/反引号等非域名典型符号
        if any(ch in url for ch in '()_"\'`'):
            return True
        host = no_proto.split('/')[0]
        parts = host.split('.')
        if len(parts) >= 3 and any(re.search(r'[A-Z]', p) for p in parts[1:-1]):
            return True
        # 过长的单段（>63）或总长>253 视为伪
        if any(len(p) > 63 for p in parts) or len(no_proto) > 253:
            return True
        # path-only 带扩展且无协议且不含点的域前缀： 已在上层过滤，这里兜底
        if '://' not in url and '/' not in url and '.' in url:
            tld = parts[-1].lower()
            if tld not in self.allowed_tlds:
                return True
        return False


# 全局实例
extractor = CodeAndLinkExtractor()