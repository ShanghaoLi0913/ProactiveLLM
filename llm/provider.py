import os
from pathlib import Path
from typing import List, Dict

from openai import OpenAI

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
def chat_complete(system_prompt: str, user_prompt: str, model: str = "gpt-4o-mini", max_tokens: int = 512) -> str:
    client = get_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.7,
    )
    return resp.choices[0].message.content or ""
# OpenAI 的聊天接口每次请求默认会返回一个或多个回复（choices），这里一般只取第一个（resp.choices[0]），因为通常只需要一条回复。
# 如果你设置 n>1，会返回多个回复选项；但我们只返回 resp.choices[0].message.content（第一个回复的内容）。
# 之所以有 "or \"\""，是因为在极少数情况下 content 可能为 None（比如接口异常或回复为空），为了避免出错，返回 ""作为默认值。