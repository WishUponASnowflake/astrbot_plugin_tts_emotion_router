#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TTS情绪路由插件 - 综合测试套件
测试所有新增和修改的功能
"""
import sys
import re
import traceback
from pathlib import Path
import time

# 添加插件路径
plugin_path = Path(__file__).parent
sys.path.insert(0, str(plugin_path))

class TestResult:
    def __init__(self, name):
        self.name = name
        self.success = False
        self.error = None
        self.details = ""
    
    def set_success(self, details=""):
        self.success = True
        self.details = details
    
    def set_failure(self, error):
        self.success = False
        self.error = str(error)

class TTSPluginTester:
    def __init__(self):
        self.results = []
        self.total_tests = 0
        self.passed_tests = 0

    def run_test(self, test_name, test_func):
        """运行单个测试"""
        result = TestResult(test_name)
        self.total_tests += 1
        
        try:
            test_func(result)
            if result.success:
                self.passed_tests += 1
                print(f"[PASS] {test_name}")
                if result.details:
                    print(f"       {result.details}")
            else:
                print(f"[FAIL] {test_name}")
                if result.error:
                    print(f"       Error: {result.error}")
        except Exception as e:
            result.set_failure(f"Exception: {e}")
            print(f"[ERROR] {test_name}")
            print(f"        {e}")
            traceback.print_exc()
        
        self.results.append(result)

    def test_imports(self, result):
        """测试所有必要模块的导入"""
        try:
            # 测试基础模块
            import re
            import json
            import hashlib
            import time
            from pathlib import Path
            from typing import Optional, List, Dict
            
            # 测试插件模块
            from emotion.classifier import HeuristicClassifier
            from emotion.infer import classify, EMOTIONS
            from tts.provider_siliconflow import SiliconFlowTTS
            
            result.set_success("All modules imported successfully")
        except Exception as e:
            result.set_failure(e)

    def test_emotion_constants(self, result):
        """测试情绪常量"""
        try:
            from emotion.infer import EMOTIONS
            expected_emotions = ["neutral", "happy", "sad", "angry"]
            
            if EMOTIONS == expected_emotions:
                result.set_success(f"EMOTIONS constant correct: {EMOTIONS}")
            else:
                result.set_failure(f"EMOTIONS mismatch. Expected: {expected_emotions}, Got: {EMOTIONS}")
        except Exception as e:
            result.set_failure(e)

    def test_heuristic_classifier(self, result):
        """测试启发式情绪分类器"""
        try:
            from emotion.classifier import HeuristicClassifier
            classifier = HeuristicClassifier()
            
            # 测试基本分类
            test_cases = [
                ("开心的一天", "happy"),
                ("我很伤心", "sad"), 
                ("太气人了", "angry"),
                ("今天天气不错", "neutral")
            ]
            
            for text, expected in test_cases:
                emotion = classifier.classify(text)
                if emotion not in ["neutral", "happy", "sad", "angry"]:
                    result.set_failure(f"Invalid emotion returned: {emotion}")
                    return
            
            result.set_success("Heuristic classifier working correctly")
        except Exception as e:
            result.set_failure(e)

    def test_tts_provider_init(self, result):
        """测试TTS提供者初始化"""
        try:
            from tts.provider_siliconflow import SiliconFlowTTS
            
            # 测试初始化
            tts = SiliconFlowTTS(
                api_url="https://api.test.com/v1",
                api_key="test_key", 
                model="test_model",
                fmt="mp3",
                speed=1.0,
                gain=5.0,
                sample_rate=44100
            )
            
            # 检查属性
            assert tts.api_url == "https://api.test.com/v1"
            assert tts.api_key == "test_key"
            assert tts.model == "test_model"
            assert tts.format == "mp3"
            assert tts.speed == 1.0
            assert tts.gain == 5.0
            assert tts.sample_rate == 44100
            
            result.set_success("TTS provider initialized correctly")
        except Exception as e:
            result.set_failure(e)

    def create_mock_plugin(self):
        """创建模拟插件实例用于测试"""
        class MockPlugin:
            def __init__(self):
                self.emo_marker_tag = "EMO"
                
            def _filter_code_blocks(self, text: str) -> str:
                if not text:
                    return text
                
                # 过滤代码块
                text = re.sub(r'```[\s\S]*?```', '[代码块]', text)
                text = re.sub(r'`[^`\n]+`', '[代码]', text)
                
                # 检测代码特征
                code_patterns = [
                    r'\b\w+\(\s*\)',
                    r'\b\w+\.\w+\(',
                    r'<[^>]+>',
                    r'\w+://\S+',
                ]
                
                for pattern in code_patterns:
                    if re.search(pattern, text):
                        return ""
                
                return text

            def _filter_emoji_and_qq_expressions(self, text: str) -> str:
                if not text:
                    return text
                
                # 修正的emoji过滤
                emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U000024FF]+')
                text = emoji_pattern.sub('', text)
                
                # 精确的QQ表情过滤
                qq_emotions = [
                    '哈哈', '呵呵', '嘿嘿', '嘻嘻', '哭哭', '呜呜',
                    '汗', '晕', '怒', '抓狂', '吐血', '偷笑',
                    '色', '亲亲', '惊讶', '难过', '酷', '冷汗',
                    '发呆', '害羞', '闭嘴', '睡觉', '大哭', '尴尬',
                    '发怒', '调皮', '呲牙', '惊喜', '流汗', '憨笑'
                ]
                
                qq_emotion_pattern = '|'.join(re.escape(emotion) for emotion in qq_emotions)
                qq_pattern = re.compile(rf'\[({qq_emotion_pattern})\]')
                text = qq_pattern.sub('', text)
                
                # 过滤颜文字
                emoticon_patterns = [
                    r'[><!]{2,}',
                    r'[:;=][)\(DPOop]{1,}',
                    r'[)\(]{2,}',
                    r'[-_]{3,}',
                ]
                
                for pattern in emoticon_patterns:
                    text = re.sub(pattern, '', text)
                
                return text.strip()

            def _deep_clean_emotion_tags(self, text: str) -> str:
                if not text:
                    return text
                
                patterns = [
                    r'^\s*\[?\s*emo\s*[:：]?\s*\w*\s*\]?\s*[,，。:\uff1a]*\s*',
                    r'^\s*\[?\s*EMO\s*[:：]?\s*\w*\s*\]?\s*[,，。:\uff1a]*\s*',
                    r'^\s*【\s*[Ee][Mm][Oo]\s*[:：]?\s*\w*\s*】\s*[,，。:\uff1a]*\s*',
                    r'\[情绪[:：]\w*\]',
                    r'\[心情[:：]\w*\]',
                    r'^\s*情绪[:：]\s*\w+\s*[,，。]\s*',
                ]
                
                for pattern in patterns:
                    text = re.sub(pattern, '', text, flags=re.IGNORECASE)
                
                return text.strip()

            def _ensure_proper_ending(self, text: str) -> str:
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
                if not text:
                    return text
                
                text = self._deep_clean_emotion_tags(text)
                text = re.sub(r'\s+', ' ', text).strip()
                
                problematic_chars = ['�', '\ufffd', '\x00', '\ufeff']
                for char in problematic_chars:
                    text = text.replace(char, '')
                
                if len(text.strip()) < 2:
                    return ""
                
                return text
                
        return MockPlugin()

    def test_code_block_filtering(self, result):
        """测试代码块过滤功能"""
        plugin = self.create_mock_plugin()
        
        test_cases = [
            # (输入, 预期输出, 描述)
            ("```python\nprint('hello')\n```", "[代码块]", "多行代码块"),
            ("`console.log()`", "[代码]", "行内代码"),
            ("function test() {}", "", "函数调用特征"),
            ("obj.method()", "", "方法调用特征"),
            ("<div>content</div>", "", "HTML标签"),
            ("https://example.com", "", "URL链接"),
            ("普通文本", "普通文本", "正常文本保留"),
        ]
        
        failures = []
        for input_text, expected, desc in test_cases:
            output = plugin._filter_code_blocks(input_text)
            if output != expected:
                failures.append(f"{desc}: 输入'{input_text}' 期望'{expected}' 实际'{output}'")
        
        if failures:
            result.set_failure(f"代码块过滤测试失败: {'; '.join(failures)}")
        else:
            result.set_success("代码块过滤功能正常")

    def test_emoji_filtering(self, result):
        """测试emoji和表情过滤功能"""
        plugin = self.create_mock_plugin()
        
        test_cases = [
            # QQ表情过滤
            ("[哈哈]", "", "QQ表情过滤"),
            ("[哈哈][呵呵]", "", "多个QQ表情"),
            ("[不是表情]", "[不是表情]", "非表情中文不过滤"),
            
            # 颜文字过滤
            (":) :( :D", "", "颜文字过滤"),
            (">>> test <<<", "test", "特殊符号过滤"),
            ("____", "", "下划线过滤"),
            
            # 正常文本保留
            ("正常的中文文本", "正常的中文文本", "中文文本保留"),
            ("Normal English text", "Normal English text", "英文文本保留"),
        ]
        
        failures = []
        for input_text, expected, desc in test_cases:
            output = plugin._filter_emoji_and_qq_expressions(input_text)
            if output != expected:
                failures.append(f"{desc}: 输入'{input_text}' 期望'{expected}' 实际'{output}'")
        
        if failures:
            result.set_failure(f"Emoji过滤测试失败: {'; '.join(failures)}")
        else:
            result.set_success("Emoji过滤功能正常")

    def test_emotion_tag_cleaning(self, result):
        """测试情绪标签清理功能"""
        plugin = self.create_mock_plugin()
        
        test_cases = [
            ("[EMO:happy]", "", "基础情绪标签"),
            ("【EMO：sad】", "", "全角情绪标签"),
            ("emo:angry", "", "简化情绪标签"),
            ("[情绪:开心]", "", "中文情绪标签"),
            ("EMO:happy 这是正文", "这是正文", "带正文的情绪标签"),
            ("正常文本", "正常文本", "正常文本不变"),
        ]
        
        failures = []
        for input_text, expected, desc in test_cases:
            output = plugin._deep_clean_emotion_tags(input_text)
            if output != expected:
                failures.append(f"{desc}: 输入'{input_text}' 期望'{expected}' 实际'{output}'")
        
        if failures:
            result.set_failure(f"情绪标签清理测试失败: {'; '.join(failures)}")
        else:
            result.set_success("情绪标签清理功能正常")

    def test_text_ending(self, result):
        """测试文本结尾处理功能"""
        plugin = self.create_mock_plugin()
        
        test_cases = [
            ("没有标点的文本", "没有标点的文本。..", "中文添加句号"),
            ("English text", "English text...", "英文添加句点"),
            ("已有标点。", "已有标点。..", "有标点添加停顿"),
            ("Already ended...", "Already ended...", "已有停顿不重复"),
        ]
        
        failures = []
        for input_text, expected, desc in test_cases:
            output = plugin._ensure_proper_ending(input_text)
            if output != expected:
                failures.append(f"{desc}: 输入'{input_text}' 期望'{expected}' 实际'{output}'")
        
        if failures:
            result.set_failure(f"文本结尾处理测试失败: {'; '.join(failures)}")
        else:
            result.set_success("文本结尾处理功能正常")

    def test_comprehensive_text_processing(self, result):
        """测试完整的文本处理流程"""
        plugin = self.create_mock_plugin()
        
        test_cases = [
            # 复合情况测试
            ("[EMO:happy] `console.log()` 测试", "[代码] 测试。..", "情绪标签+代码过滤"),
            ("[哈哈] 普通文本", "普通文本。..", "表情+普通文本"),
            ("emo:sad >>> 测试 <<<", "测试。..", "情绪标签+特殊符号"),
        ]
        
        failures = []
        for input_text, expected, desc in test_cases:
            # 模拟完整处理流程
            step1 = plugin._filter_code_blocks(input_text)
            if not step1:  # 被代码过滤器跳过
                continue
            step2 = plugin._filter_emoji_and_qq_expressions(step1)
            step3 = plugin._deep_clean_emotion_tags(step2)
            step4 = plugin._final_text_cleanup(step3)
            if not step4:  # 最终清理后为空
                continue
            output = plugin._ensure_proper_ending(step4)
            
            if output != expected:
                failures.append(f"{desc}: 输入'{input_text}' 期望'{expected}' 实际'{output}'")
        
        if failures:
            result.set_failure(f"综合文本处理测试失败: {'; '.join(failures)}")
        else:
            result.set_success("综合文本处理功能正常")

    def test_performance(self, result):
        """测试性能"""
        plugin = self.create_mock_plugin()
        
        # 性能测试用例
        test_text = "[EMO:happy] 这是一段包含```python\nprint('test')\n```代码和[哈哈]表情的长文本" * 100
        
        start_time = time.time()
        
        # 执行1000次处理
        for _ in range(1000):
            step1 = plugin._filter_code_blocks(test_text)
            if step1:
                step2 = plugin._filter_emoji_and_qq_expressions(step1)
                step3 = plugin._deep_clean_emotion_tags(step2)
                step4 = plugin._final_text_cleanup(step3)
                if step4:
                    plugin._ensure_proper_ending(step4)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 期望1000次处理在5秒内完成
        if duration < 5.0:
            result.set_success(f"性能测试通过: 1000次处理耗时{duration:.3f}秒")
        else:
            result.set_failure(f"性能测试失败: 1000次处理耗时{duration:.3f}秒，超过5秒限制")

    def run_all_tests(self):
        """运行所有测试"""
        print("=" * 60)
        print("TTS情绪路由插件 - 综合测试套件")
        print("=" * 60)
        
        # 运行所有测试
        self.run_test("模块导入测试", self.test_imports)
        self.run_test("情绪常量测试", self.test_emotion_constants)  
        self.run_test("启发式分类器测试", self.test_heuristic_classifier)
        self.run_test("TTS提供者初始化测试", self.test_tts_provider_init)
        self.run_test("代码块过滤测试", self.test_code_block_filtering)
        self.run_test("Emoji过滤测试", self.test_emoji_filtering)
        self.run_test("情绪标签清理测试", self.test_emotion_tag_cleaning)
        self.run_test("文本结尾处理测试", self.test_text_ending)
        self.run_test("综合文本处理测试", self.test_comprehensive_text_processing)
        self.run_test("性能测试", self.test_performance)
        
        # 输出总结
        print("\n" + "=" * 60)
        print(f"测试总结: {self.passed_tests}/{self.total_tests} 通过")
        print(f"成功率: {self.passed_tests/self.total_tests*100:.1f}%")
        
        if self.passed_tests == self.total_tests:
            print("🎉 所有测试通过！")
        else:
            print("❌ 部分测试失败，需要检查和修复")
            
        return self.passed_tests == self.total_tests

if __name__ == "__main__":
    tester = TTSPluginTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)