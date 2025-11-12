# =============================================================================
# 文件: llm/test_openai_key.py
#
# 作用说明:
# -----------------------------------------------------------------------------
# 本脚本用于检测 OpenAI API key 是否正确配置。它包括如下检测步骤:
#   1. 检查环境变量 OPENAI_API_KEY 是否已设置，并提示设置方法；
#   2. 校验 key 格式是否合理（应以 "sk-" 开头，并做简要输出提醒）；
#   3. 实际调用一次 OpenAI API（对话接口），验证密钥是否可正常请求服务与返回；
# 
# 若所有检查通过，会输出详细提示（可安全使用）；如有问题，将清楚提示错误原因与排查建议。
# =============================================================================

import os
import sys
from pathlib import Path

# 添加项目根目录到路径，以便导入llm模块
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from llm.provider import get_client, chat_complete

def test_api_key():
    """测试API key是否可以正常使用"""
    print("=" * 60)
    print("OpenAI API Key 测试")
    print("=" * 60)
    
    # 检查环境变量
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ 错误: OPENAI_API_KEY 环境变量未设置")
        print("\n请按以下方式设置API key:")
        print("  方式1 (临时，当前终端会话):")
        print("    export OPENAI_API_KEY='sk-...'")
        print("\n  方式2 (永久，添加到 ~/.bashrc 或 ~/.zshrc):")
        print("    echo 'export OPENAI_API_KEY=\"sk-...\"' >> ~/.bashrc")
        print("    source ~/.bashrc")
        print("\n  方式3 (当前会话，直接运行):")
        print("    OPENAI_API_KEY='sk-...' python llm/test_openai_key.py")
        return False
    
    # 检查格式
    if not api_key.startswith('sk-'):
        print(f"⚠️  警告: API key格式可能不正确 (应以'sk-'开头)")
        print(f"   当前key前10个字符: {api_key[:10]}...")
    else:
        print(f"✓ API key已设置 (格式正确)")
        print(f"  Key前缀: {api_key[:7]}...")
    
    # 测试API调用
    print("\n测试API连接...")
    try:
        client = get_client()
        print("✓ Client创建成功")
        
        # 简单测试调用
        print("测试API调用...")
        response = chat_complete(
            system_prompt="You are a helpful assistant.",
            user_prompt="Say 'Hello' in one word.",
            model="gpt-4o-mini",
            max_tokens=10
        )
        print(f"✓ API调用成功!")
        print(f"  响应: {response.strip()}")
        print("\n" + "=" * 60)
        print("✅ OpenAI API key 配置正确，可以正常使用!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"❌ API调用失败: {e}")
        print("\n可能的原因:")
        print("  1. API key无效或已过期")
        print("  2. API key没有足够的权限")
        print("  3. 网络连接问题")
        print("  4. OpenAI服务暂时不可用")
        return False

if __name__ == "__main__":
    success = test_api_key()
    sys.exit(0 if success else 1)

