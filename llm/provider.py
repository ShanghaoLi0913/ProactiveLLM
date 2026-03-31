import os
import sys
import time
from pathlib import Path
from typing import List, Dict

from openai import OpenAI
from openai import APIConnectionError, APITimeoutError, RateLimitError, InternalServerError, BadRequestError

# 普通的 from import 导入模块语句（from dotenv import load_dotenv）。
# 这里用 try...except 包裹，是为了兼容调试环境防止没有安装 python-dotenv 时报错影响主程序流程。
# 这样写法的好处是：如果你没装 python-dotenv，不会导致整个程序报错，只是跳过自动加载 .env 文件的这一步，
# 主要便于本地开发时自动加载密钥，也方便部署环境灵活切换。
try:
    from dotenv import load_dotenv
    # 加载项目根目录的 .env 文件（方便本地开发自动获取密钥）
    project_root = Path(__file__).resolve().parent.parent
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
except ImportError:
    # 如果没有安装 python-dotenv，则不会中断程序，只是跳过 .env 文件自动加载（环境变量需要你自己提前设好）。
    # except ImportError 是标准Python语法。它用于捕获导入模块时发生的 ImportError 异常。
    pass

# 这部分代码主要有两个函数：get_client 和 chat_complete，其中 get_client 函数用于创建和返回一个 OpenAI 客户端实例，
# chat_complete 函数用于以对话形式调用 OpenAI 的聊天接口，让大模型根据你的提示词生成回复。

# 1. get_client() 函数：
#    - 作用：用于创建和返回一个 OpenAI 客户端实例。
#    - 逻辑：
#        a. 首先从环境变量中读取 "OPENAI_API_KEY"（你的 OpenAI 密钥）。
#        b. 如果没有设置 key，会抛出一个异常，提醒你要配置 API key（可以在环境变量或 .env 文件里设置）。
#        c. 如果 key 存在，返回一个 OpenAI 客户端对象，后续用于调用 OpenAI API。

# 对，这里的"-> OpenAI"只是类型提示，不会被执行，对运行没有影响。它只是告诉读者/IDE/类型检查工具“这个函数返回OpenAI类型”，提高可读性，不参与任何业务逻辑。
def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set. Please set it as environment variable or in .env file")
    return OpenAI(api_key=api_key)

# 2. chat_complete(...) 函数：
#    - 作用：以对话形式调用 OpenAI 的聊天接口，让大模型根据你的提示词生成回复。
#    - 逻辑：
#        a. 先通过 get_client() 拿到 OpenAI 客户端（确保有 API key）。
#        b. 使用 client.chat.completions.create(...) 方法发起聊天补全请求，
#           - 参数包括：模型名（默认 "gpt-4o-mini"）、system_prompt、user_prompt、最大 tokens、temperature（采样温度）。
#           - system_prompt 通常定义“助手的角色和行为”；
#           - user_prompt 是用户具体想让助手处理的内容。
#        c. 返回模型生成的回复文本（只取第一个候选）。
def chat_complete(
    system_prompt: str,
    user_prompt: str,
    model: str = "gpt-4o-mini",
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float | None = None,
    timeout_s: float = 60.0,
    max_retries: int = 12,  # Increased from 8 to 12 for better resilience
) -> str:
    client = get_client()
    kwargs = {}
    # OpenAI Chat Completions supports both temperature and top_p; keep optional for backward compatibility
    if top_p is not None:
        # Validate top_p range [0, 1]
        if not (0.0 <= top_p <= 1.0):
            print(f"⚠️  Invalid top_p={top_p}, clamping to [0, 1]", file=sys.stderr, flush=True)
            top_p = max(0.0, min(1.0, top_p))
        kwargs["top_p"] = top_p
    
    # Validate temperature range [0, 2]
    if not (0.0 <= temperature <= 2.0):
        print(f"⚠️  Invalid temperature={temperature}, clamping to [0, 2]", file=sys.stderr, flush=True)
        temperature = max(0.0, min(2.0, temperature))
    
    # Validate and sanitize prompts (remove None, ensure strings)
    if system_prompt is None:
        system_prompt = ""
    if user_prompt is None:
        user_prompt = ""
    
    # Ensure prompts are strings and not too long
    system_prompt = str(system_prompt)[:100000]  # Limit to 100k chars
    user_prompt = str(user_prompt)[:100000]  # Limit to 100k chars

    # Retry transient network/429/5xx issues (keeps long-running data-gen jobs alive).
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout_s,
                **kwargs,
            )
            return resp.choices[0].message.content or ""
        except (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError, BadRequestError) as e:
            last_err = e
            # Exponential backoff with cap: longer wait for connection errors
            if isinstance(e, APIConnectionError):
                # For connection errors, wait longer (up to 60 seconds)
                sleep_s = min(60.0, 2.0 ** attempt)
            elif isinstance(e, BadRequestError):
                # For bad requests, don't retry too many times (likely a permanent issue)
                if attempt >= 3:  # Only retry 3 times for bad requests
                    print(f"❌ BadRequestError after {attempt} attempts, likely a permanent issue (invalid prompt/parameters)", file=sys.stderr, flush=True)
                    raise e
                sleep_s = min(10.0, 2.0 ** attempt)  # Shorter wait for bad requests
            else:
                # For other errors, use shorter backoff
                sleep_s = min(30.0, 1.5 ** attempt)
            
            if attempt < max_retries:
                error_msg = str(e)[:200] if str(e) else type(e).__name__
                print(f"⚠️  API调用失败 (尝试 {attempt}/{max_retries}): {type(e).__name__}: {error_msg}, 等待 {sleep_s:.1f}秒后重试...", file=sys.stderr, flush=True)
            time.sleep(sleep_s)

    # Non-transient or exceeded retries
    if last_err is not None:
        print(f"❌ API调用失败，已重试 {max_retries} 次: {type(last_err).__name__}: {last_err}", file=sys.stderr, flush=True)
        raise last_err
    return ""
# OpenAI 的聊天接口每次请求默认会返回一个或多个回复（choices），这里一般只取第一个（resp.choices[0]），因为通常只需要一条回复。
# 如果你设置 n>1，会返回多个回复选项；但我们只返回 resp.choices[0].message.content（第一个回复的内容）。
# 之所以有 "or \"\""，是因为在极少数情况下 content 可能为 None（比如接口异常或回复为空），为了避免出错，返回 ""作为默认值。