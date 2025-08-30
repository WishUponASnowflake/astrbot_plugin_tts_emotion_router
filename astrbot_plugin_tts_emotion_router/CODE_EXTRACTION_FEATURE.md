# 代码和链接提取功能说明

## 功能概述

这个功能改进了 TTS 情绪路由插件，实现了以下特性：

1. **智能识别代码**：能够区分真正的代码块和技术术语（如模型名），避免将 GPT-4、Claude-3 等模型名误识别为代码
2. **提取代码和链接**：从文本中提取代码块和链接，生成参考文献格式
3. **优化 TTS 输出**：清理文本中的代码和链接，只保留需要朗读的内容
4. **参考文献展示**：当检测到代码或链接时，额外发送一条非 TTS 的文本消息，以参考文献格式展示

## 实现细节

### 1. 新增文件
- `utils/extract.py`：代码和链接提取的核心逻辑
- `test_extract.py`：测试文件
- `CODE_EXTRACTION_FEATURE.md`：功能说明文档

### 2. 修改的文件
- `main.py`：集成提取功能到 TTS 处理流程
- `emotion/infer.py`：改进代码识别逻辑

### 3. 核心功能

#### 代码和链接提取
- 支持多行代码块（```language\n...\n```）
- 支持行内代码（`code`）
- 支持链接（https:// 和 www.）
- 智能过滤技术术语（如 GPT-4、Claude、API 等）

#### 文本清理
- 移除代码块和链接
- 保留技术术语
- 清理多余空白

#### 参考文献格式
```
📚 参考文献和代码片段：
[代码1] python代码片段: def hello(): print("Hello, World!")...
[链接1] https://example.com
```

### 4. 使用场景

#### 场景1：技术讨论
输入：
```
我推荐使用 GPT-4 模型。示例代码：
```python
import openai
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}]
)
```
详情见：https://platform.openai.com/docs
```

TTS 会朗读：
```
我推荐使用 GPT-4 模型。示例代码：
```

额外发送的文本：
```
我推荐使用 GPT-4 模型。示例代码：

📚 参考文献和代码片段：
[代码1] python代码片段: import openai response = openai.ChatCompletion.create(...
[链接1] https://platform.openai.com/docs
详情见：
```

#### 场景2：模型推荐
输入：
```
GPT-4 和 Claude-3 都是很好的大语言模型，你可以根据需求选择。
```

TTS 会朗读完整文本，不会误识别为代码。

## 配置

无需额外配置，功能自动启用。当 `allow_mixed` 开启时，参考文献会附加在文本消息后；当 `allow_mixed` 关闭时，只会发送音频，但代码和链接仍会被过滤。

## 注意事项

1. 只有真正的代码块会被提取，单个技术术语会被保留
2. 行内代码只有在包含复杂内容时才会被提取
3. 参考文献只在检测到代码或链接时才会生成
4. 原有的 TTS 功能（情绪路由、音色切换等）保持不变