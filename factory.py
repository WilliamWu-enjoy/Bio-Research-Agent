
from langchain_openai import ChatOpenAI
from config import model_config

# 加载 .env 文件中的环境变量到系统的 os.environ 中
# 这样代码就能隐式地读到了，而不需要明文写在这里



def get_chat_model():
    """
    初始化并返回大语言模型 (对应架构图中的 chat_model)
    """
    # 提醒：实际开发中，千万不要把真实的 API Key 明文写在代码里传到 GitHub 上！



    if not model_config.api_key:
        raise ValueError("哎呀，没有找到 API KEY呢！")
    # 实例化大模型
    llm = ChatOpenAI(
        model=model_config.model_name,  # 如果用国内模型，这里换成对应的名字，比如 "deepseek-chat"
        temperature=model_config.temperature,  # 重点面试考点：temperature 控制回答的随机性。
        # 对于我们这个严谨的生物科研智能体，我们需要准确的 Protocol 和试剂配比，
        # 绝对不能让它天马行空地发挥，所以要把 temperature 调低（接近 0）。
        api_key=model_config.api_key,
        base_url=model_config.base_url,
        streaming=model_config.streaming  # 开启流式输出，配合前端的打字机效果
    )

    return llm
