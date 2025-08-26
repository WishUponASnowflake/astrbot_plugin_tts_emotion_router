#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
&符号情绪标签过滤测试脚本
专门测试&标签&格式的情绪标签过滤功能
"""
import sys
import re
from pathlib import Path

# 添加插件路径
plugin_path = Path(__file__).parent
sys.path.insert(0, str(plugin_path))

class AmpersandEmotionTagTest:
    def __init__(self):
        self.passed = 0
        self.total = 0

    def create_mock_plugin(self):
        """创建模拟插件用于测试"""
        class MockPlugin:
            def _deep_clean_emotion_tags(self, text: str) -> str:
                """深度清理各种形式的情绪标签"""
                if not text:
                    return text
                
                # 与实际插件相同的模式
                patterns = [
                    r'^\s*\[?\s*emo\s*[:：]?\s*\w*\s*\]?\s*[,，。:\uff1a]*\s*',
                    r'^\s*\[?\s*EMO\s*[:：]?\s*\w*\s*\]?\s*[,，。:\uff1a]*\s*',
                    r'^\s*【\s*[Ee][Mm][Oo]\s*[:：]?\s*\w*\s*】\s*[,，。:\uff1a]*\s*',
                    r'\[情绪[:：]\w*\]',
                    r'\[心情[:：]\w*\]',
                    r'^\s*情绪[:：]\s*\w+\s*[,，。]\s*',
                    
                    # 新增：【情绪：xxx】格式支持
                    r'【情绪[:：][^】]*】',     # 【情绪：开心】等全角格式
                    r'【心情[:：][^】]*】',     # 【心情：开心】等全角格式
                    
                    # 新增：&符号包围的情绪标签
                    r'&[a-zA-Z\u4e00-\u9fff]+&',  # &英文或中文&，匹配任意位置
                    r'^\s*&[a-zA-Z\u4e00-\u9fff]+&\s*[,，。:\uff1a]*\s*',  # 开头的&标签&带标点
                ]
                
                for pattern in patterns:
                    text = re.sub(pattern, '', text, flags=re.IGNORECASE)
                
                return text.strip()

            def _filter_emoji_and_qq_expressions(self, text: str) -> str:
                """简化的emoji过滤"""
                if not text:
                    return text
                # 基本的emoji过滤（简化版本）
                return text

            def _ensure_proper_ending(self, text: str) -> str:
                """确保适当的结尾"""
                if not text or not text.strip():
                    return text
                
                text = text.strip()
                
                if not re.search(r'[。！？.!?，,]$', text):
                    if re.search(r'[\u4e00-\u9fff]', text):
                        text += '。'
                    else:
                        text += '.'
                
                if not text.endswith('...'):
                    text += '..'
                
                return text

            def _final_text_cleanup(self, text: str) -> str:
                """最终文本清理"""
                if not text:
                    return text
                
                text = self._deep_clean_emotion_tags(text)
                text = re.sub(r'\s+', ' ', text).strip()
                
                if len(text.strip()) < 2:
                    return ""
                
                return text

            def process_full(self, text):
                """完整处理流程"""
                step1 = self._filter_emoji_and_qq_expressions(text)
                step2 = self._deep_clean_emotion_tags(step1)
                step3 = self._final_text_cleanup(step2)
                if not step3:
                    return ""
                return self._ensure_proper_ending(step3)

        return MockPlugin()

    def run_test(self, name, test_func):
        """运行单个测试"""
        self.total += 1
        try:
            result = test_func()
            if result:
                self.passed += 1
                print(f"[PASS] {name}")
            else:
                print(f"[FAIL] {name}")
        except Exception as e:
            print(f"[ERROR] {name}: {e}")

    def test_basic_ampersand_tags(self):
        """测试基本的&标签过滤"""
        plugin = self.create_mock_plugin()
        
        test_cases = [
            # (输入, 预期输出或描述, 描述)
            ("&shy& 这是害羞的表情", "这是害羞的表情。..", "&shy&标签过滤"),
            ("&开心& 今天天气很好", "今天天气很好。..", "中文&开心&标签"),
            ("&happy& 测试英文情绪", "测试英文情绪。..", "英文&happy&标签"),
            ("&angry& 很生气的消息", "很生气的消息。..", "&angry&标签"),
            ("&neutral& 中性情绪测试", "中性情绪测试。..", "&neutral&标签"),
        ]
        
        failures = []
        for input_text, expected, desc in test_cases:
            result = plugin.process_full(input_text)
            if result != expected:
                failures.append(f"{desc}: 输入'{input_text}' 期望'{expected}' 实际'{result}'")
        
        if failures:
            print(f"       失败详情: {'; '.join(failures)}")
            return False
        return True

    def test_middle_position_tags(self):
        """测试中间位置的&标签"""
        plugin = self.create_mock_plugin()
        
        test_cases = [
            ("前面文字 &sad& 后面文字", "前面文字 后面文字。..", "中间位置&sad&"),
            ("开始 &joy& 中间 &worry& 结束", "开始 中间 结束。..", "多个&标签&"),
            ("text &confused& more text", "text more text...", "英文中间&标签&"),
        ]
        
        failures = []
        for input_text, expected, desc in test_cases:
            result = plugin.process_full(input_text)
            if result != expected:
                failures.append(f"{desc}: 输入'{input_text}' 期望'{expected}' 实际'{result}'")
        
        if failures:
            print(f"       失败详情: {'; '.join(failures)}")
            return False
        return True

    def test_false_positive_protection(self):
        """测试防止误删正常的&符号使用"""
        plugin = self.create_mock_plugin()
        
        test_cases = [
            ("R&D部门很忙", "R&D部门很忙。..", "R&D不被误删"),
            ("A&B两个选项", "A&B两个选项。..", "A&B不被误删"),
            ("Tom & Jerry", "Tom & Jerry...", "Tom & Jerry不被误删"),
            ("价格是100&以上", "价格是100&以上。..", "数字&不被误删"),
            ("H&M品牌", "H&M品牌。..", "品牌名&不被误删"),
        ]
        
        failures = []
        for input_text, expected, desc in test_cases:
            result = plugin.process_full(input_text)
            if result != expected:
                failures.append(f"{desc}: 输入'{input_text}' 期望'{expected}' 实际'{result}'")
        
        if failures:
            print(f"       失败详情: {'; '.join(failures)}")
            return False
        return True

    def test_edge_cases(self):
        """测试边缘情况"""
        plugin = self.create_mock_plugin()
        
        test_cases = [
            ("&& 空标签", "&& 空标签。..", "空&标签&不匹配"),
            ("&123& 数字标签", "&123& 数字标签。..", "数字&标签&不匹配"),  
            ("&a& 单字母", "单字母。..", "单字母&标签&匹配"),
            ("&verylongword& 长词", "长词。..", "长词&标签&匹配"),
            ("&mixed123& 混合", "&mixed123& 混合。..", "混合字符&标签&不匹配数字"),
        ]
        
        failures = []
        for input_text, expected, desc in test_cases:
            result = plugin.process_full(input_text)
            if result != expected:
                failures.append(f"{desc}: 输入'{input_text}' 期望'{expected}' 实际'{result}'")
        
        if failures:
            print(f"       失败详情: {'; '.join(failures)}")
            return False
        return True

    def test_mixed_formats(self):
        """测试混合多种标签格式"""
        plugin = self.create_mock_plugin()
        
        test_cases = [
            ("[EMO:happy] &sad& 混合标签", "混合标签。..", "EMO和&标签混合"),
            ("&angry& emo:neutral 多种格式", "多种格式。..", "多种格式混合"),
            ("【情绪：开心】&shy& 全部清理", "全部清理。..", "所有格式清理"),
        ]
        
        failures = []
        for input_text, expected, desc in test_cases:
            result = plugin.process_full(input_text)
            if result != expected:
                failures.append(f"{desc}: 输入'{input_text}' 期望'{expected}' 实际'{result}'")
        
        if failures:
            print(f"       失败详情: {'; '.join(failures)}")
            return False
        return True

    def test_regex_patterns_directly(self):
        """直接测试正则表达式模式"""
        # 测试&标签的正则模式
        pattern1 = r'&[a-zA-Z\u4e00-\u9fff]+&'
        pattern2 = r'^\s*&[a-zA-Z\u4e00-\u9fff]+&\s*[,，。:\uff1a]*\s*'
        
        test_cases = [
            # (文本, 模式, 应该匹配)
            ("&shy&", pattern1, True),
            ("&开心&", pattern1, True),
            ("&happy&", pattern1, True),
            ("&123&", pattern1, False),  # 数字不匹配
            ("R&D", pattern1, False),    # 单个&不匹配
            ("&a&", pattern1, True),     # 单字母匹配
            ("&test123&", pattern1, False), # 混合数字不匹配
            ("&test_word&", pattern1, False), # 下划线不匹配
        ]
        
        failures = []
        for text, pattern, should_match in test_cases:
            matches = bool(re.search(pattern, text))
            if matches != should_match:
                failures.append(f"'{text}' vs pattern: 期望{should_match} 实际{matches}")
        
        if failures:
            print(f"       正则测试失败: {'; '.join(failures)}")
            return False
        return True

    def run_all_tests(self):
        """运行所有测试"""
        print("=" * 60)
        print("&符号情绪标签过滤测试")
        print("=" * 60)
        
        self.run_test("基本&标签过滤测试", self.test_basic_ampersand_tags)
        self.run_test("中间位置&标签测试", self.test_middle_position_tags)
        self.run_test("防误删正常&符号测试", self.test_false_positive_protection)
        self.run_test("边缘情况测试", self.test_edge_cases)
        self.run_test("混合标签格式测试", self.test_mixed_formats)
        self.run_test("正则表达式模式测试", self.test_regex_patterns_directly)
        
        print("\n" + "=" * 60)
        print(f"测试结果: {self.passed}/{self.total} 通过")
        print(f"成功率: {self.passed/self.total*100:.1f}%")
        
        if self.passed == self.total:
            print("所有&标签过滤测试通过！")
            return True
        else:
            print("部分测试失败，需要检查")
            return False

if __name__ == "__main__":
    tester = AmpersandEmotionTagTest()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)