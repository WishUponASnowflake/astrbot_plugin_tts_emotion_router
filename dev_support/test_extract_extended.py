#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""扩展测试：验证改进后的行内代码与链接判定不会误伤普通名称 (已移至 dev_support)。"""
from pathlib import Path
import sys

plugin_dir = Path(__file__).parent.parent
if str(plugin_dir) not in sys.path:
    sys.path.insert(0, str(plugin_dir))

from utils.extract import extractor  # noqa


def run_case(name: str, text: str, expect_code: int, expect_link: int):
    extracted = extractor.extract_all(text)
    code_items = [e for e in extracted if e.type == 'code']
    link_items = [e for e in extracted if e.type == 'link']
    ok = (len(code_items) == expect_code) and (len(link_items) == expect_link)
    status = 'PASS' if ok else 'FAIL'
    print(f"[{status}] {name} -> code={len(code_items)}/{expect_code}, link={len(link_items)}/{expect_link}")
    if not ok:
        print('  Extracted detail:')
        for e in extracted:
            print('   -', e.type, e.content)
    assert ok, name


def main():
    run_case(
        '普通名称 tavily 行内反引号不算代码',
        '这是 `tavily` 的 API Key, 不应该被当成代码。',
        expect_code=0,
        expect_link=0,
    )
    run_case(
        '变量风格 TAVILY_API_KEY 识别为代码',
        '请把 `TAVILY_API_KEY` 设置到环境变量里。',
        expect_code=1,
        expect_link=0,
    )
    run_case(
        '伪域名不当作链接',
        '调用 openai.ChatCompletion.create 可以发送请求。',
        expect_code=0,
        expect_link=0,
    )
    run_case(
        '真实域名识别为链接',
        '访问 openai.com 获取更多信息。',
        expect_code=0,
        expect_link=1,
    )
    run_case(
        '重复域名去重',
        '参考 www.example.org 。再次引用 www.example.org 仍应只计一次。',
        expect_code=0,
        expect_link=1,
    )
    run_case(
        '半截文件名不当作链接',
        '请阅读 plugin.html 文档，小写扩展名不应触发链接识别。',
        expect_code=0,
        expect_link=0,
    )
    run_case(
        '纯扩展名不算代码',
        '这个文件后缀是 `.py` 用 Python 写的。',
        expect_code=0,
        expect_link=0,
    )
    run_case(
        '简单文件名不算代码',
        '第一个示例是 `hello.py` ，里面只有一个 print。',
        expect_code=0,
        expect_link=0,
    )


if __name__ == '__main__':
    main()
