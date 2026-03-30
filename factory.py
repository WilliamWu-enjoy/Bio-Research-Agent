import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量到系统的 os.environ 中
# 这样代码就能隐式地读到了，而不需要明文写在这里
load_dotenv()


def get_chat_model():
    """
    初始化并返回大语言模型 (对应架构图中的 chat_model)
    """
    # 提醒：实际开发中，千万不要把真实的 API Key 明文写在代码里传到 GitHub 上！
    # 最好通过系统的环境变量，或者 .env 文件来读取。这里为了方便测试先写死或用默认值。
    api_key = os.environ.get("OPENAI_API_KEY")

    # 如果你用的是国内模型（比如 DeepSeek），需要填上厂商提供的 base_url
    base_url = os.environ.get("OPENAI_BASE_URL")

    if not api_key:
        raise ValueError("哎呀，没有找到 API KEY呢！")
    # 实例化大模型
    llm = ChatOpenAI(
        model="Qwen/Qwen3.5-397B-A17B",  # 如果用国内模型，这里换成对应的名字，比如 "deepseek-chat"
        temperature=0.1,  # 重点面试考点：temperature 控制回答的随机性。
        # 对于我们这个严谨的生物科研智能体，我们需要准确的 Protocol 和试剂配比，
        # 绝对不能让它天马行空地发挥，所以要把 temperature 调低（接近 0）。
        api_key=api_key,
        base_url=base_url,
        streaming=True  # 开启流式输出，配合前端的打字机效果
    )

    return llm
