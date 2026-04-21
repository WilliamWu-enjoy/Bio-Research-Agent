from langchain.agents import create_agent
from factory import get_chat_model
# 确保这里的文件名和你的工具箱文件名一致（看你刚才的截图是 agent_tool）
from agent_tool import tools_list
from config import agent_config

class ReactAgent:
    def __init__(self):
        """
        初始化智能体 (基于最新架构)
        """
        # 1. 引入大脑
        self.llm = get_chat_model()

        # 2. 引入手脚
        self.tools = tools_list

        # 3. 设定系统提示词 (新版 API 非常简洁，直接传字符串即可)
        self.system_prompt = agent_config.system_prompt

        # 4. 创建 Agent
        # 新版 API：不再需要繁琐的 AgentExecutor，create_agent 会在底层自动帮你构建好循环引擎
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=self.system_prompt
        )

    def execute(self, query: str, chat_history: list = None):
        if chat_history is None:
            chat_history = []

        print(f"\n[Agent 引擎启动] 接收到任务: {query}")
        messages = chat_history + [{"role": "user", "content": query}]

        # 核心修改：在这里加入 config 配置，强制限制最多循环 5 次（防止死循环破产）
        response = self.agent.invoke(
            {"messages": messages},
            config={"recursion_limit":agent_config.recursion_limit}
        )

        return response["messages"][-1].content


# ==========================================
# 本地测试代码 (见证奇迹的时刻)
# ==========================================
if __name__ == "__main__":
    my_agent = ReactAgent()

    print("\n--- 测试任务 1 ---")
    question1 = "我要配制总体积10 mL，目标浓度为 0.1 的细胞因子工作液，我的母液浓度是 5.0，帮我算算需要取多少母液？"
    answer1 = my_agent.execute(question1)
    print(f"\n最终回答:\n{answer1}")

    print("\n--- 测试任务 2 ---")
    question2 = "帮我查一下 PubMed 上关于 'retinal organoids electrical stimulation' 的最新文献PMID。"
    answer2 = my_agent.execute(question2)
    print(f"\n最终回答:\n{answer2}")