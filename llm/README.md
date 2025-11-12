# LLM Provider 模块

## 概述

`llm/provider.py` 是项目的 OpenAI API 提供者模块，负责封装 OpenAI API 的调用。

## API 引入方式

### 在代码中使用

```python
from llm.provider import chat_complete

# 调用 OpenAI API
response = chat_complete(
    system_prompt="You are a helpful assistant.",
    user_prompt="Say hello",
    model="gpt-4o-mini",
    max_tokens=512
)
```

### API Key 配置

API key 通过环境变量 `OPENAI_API_KEY` 提供，支持两种配置方式：

#### 方式1: 环境变量（推荐）

```bash
# 临时设置（当前终端）
export OPENAI_API_KEY='sk-...'

# 永久设置（添加到 ~/.bashrc）
echo 'export OPENAI_API_KEY="sk-..."' >> ~/.bashrc
source ~/.bashrc
```

#### 方式2: .env 文件

在项目根目录创建 `.env` 文件：

```
OPENAI_API_KEY=sk-...
```

需要安装 `python-dotenv`：
```bash
pip install python-dotenv
```

---

## 本项目实际使用的配置方式

### ✅ 当前配置：环境变量（方式1 - 永久设置）

本项目**使用环境变量方式**，具体是通过**永久设置到 `~/.bashrc`** 来实现的。

**配置位置**：`~/.bashrc` 文件中包含：
```bash
export OPENAI_API_KEY='sk-...'
```

### 为什么选择这种方式？

1. **适合云服务器环境**
   - 本项目运行在 Autodl 云服务器上
   - 每次重新连接终端都需要重新设置会很麻烦
   - 永久设置到 `~/.bashrc` 可以一次配置，永久生效

2. **简单可靠**
   - 不需要额外的依赖（`.env` 方式需要 `python-dotenv`）
   - 系统级别的配置，所有 Python 脚本都能访问
   - 不需要修改项目代码，直接使用标准的环境变量机制

3. **安全性**
   - API key 不会提交到 Git（`.bashrc` 是用户配置文件）
   - 不会出现在项目代码中
   - 每个用户在自己的服务器上独立配置

4. **兼容性好**
   - 所有支持环境变量的工具都能使用
   - 不需要在代码中特殊处理 `.env` 文件加载
   - 符合标准的配置管理实践

### 代码中的实现

在 `provider.py` 中，API key 的读取逻辑：

```python
def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")  # 从环境变量读取
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set. Please set it as environment variable or in .env file")
    return OpenAI(api_key=api_key)
```

虽然代码也支持 `.env` 文件（通过 `python-dotenv`），但本项目实际使用的是环境变量方式，因为：
- 更简单，不需要额外依赖
- 更适合云服务器环境
- 配置更持久可靠

### 验证配置

运行以下命令验证 API key 是否已正确配置：

```bash
# 检查环境变量
echo $OPENAI_API_KEY

# 运行测试脚本
python llm/test_openai_key.py
```

### 工作原理

1. **加载顺序**：
   - 首先尝试从 `.env` 文件加载（如果安装了 `python-dotenv`）
   - 然后从环境变量 `OPENAI_API_KEY` 读取

2. **API 调用流程**：
   ```
   chat_complete() 
   → get_client() 
   → 读取 OPENAI_API_KEY 
   → 创建 OpenAI 客户端 
   → 调用 API 
   → 返回响应
   ```

3. **错误处理**：
   - 如果 API key 未设置，会抛出 `RuntimeError`
   - 错误信息会提示如何设置 API key

## 使用示例

### 在 scripts/generate_data.py 中的使用

```python
from llm.provider import chat_complete

def llm_output(state: Dict, action_prompt: str, model: str) -> str:
    system = action_prompt
    user = f"[Task]\n{state['query']}"
    return chat_complete(system, user, model=model, max_tokens=400)
```

## 函数说明

### `get_client() -> OpenAI`

创建并返回 OpenAI 客户端实例。

- **返回**: `OpenAI` 客户端对象
- **异常**: 如果 `OPENAI_API_KEY` 未设置，抛出 `RuntimeError`

### `chat_complete(system_prompt, user_prompt, model="gpt-4o-mini", max_tokens=512) -> str`

发送聊天完成请求。

- **参数**:
  - `system_prompt` (str): 系统提示词
  - `user_prompt` (str): 用户提示词
  - `model` (str): 模型名称，默认 `"gpt-4o-mini"`
  - `max_tokens` (int): 最大 token 数，默认 `512`
- **返回**: API 响应的文本内容
- **异常**: 如果 API 调用失败，会抛出相应的异常

## 依赖

- `openai>=1.44.0`: OpenAI Python SDK
- `python-dotenv>=1.0.0` (可选): 支持 `.env` 文件

## 测试

运行测试脚本验证 API key 配置：

```bash
python llm/test_openai_key.py
```

## 文件说明

- `provider.py`: 核心模块，提供 OpenAI API 封装
- `test_openai_key.py`: 测试脚本，用于验证 API key 配置
- `README.md`: 本文件，说明 API 引入和使用方式

