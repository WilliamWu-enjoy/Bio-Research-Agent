import streamlit as st
from react_agent import ReactAgent

# 1. 页面基本设置
st.set_page_config(
    page_title="生物科研智能体 | Bio-Research Agent",
    page_icon="🧬",
    layout="centered"
)

st.title("🧬 实验室专属科研 Agent")
st.caption("支持本地文献库检索 (RAG) 与自动工具调用 (ReAct) - 尤其擅长类器官与电生理数据分析")

# 2. 初始化核心 Agent 到网页的记忆中
# 这样保证每次用户提问时，不需要重新去加载一次大模型
if "agent" not in st.session_state:
    st.session_state.agent = ReactAgent()

# 3. 初始化聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "你好！我是你的科研辅助智能体。你可以让我**计算试剂配比**、**检索 PubMed 最新文献**，或者**查询本地文献库**中的具体实验参数。"}
    ]

# 4. 在界面上渲染之前的聊天记录
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 5. 处理用户的最新输入
if prompt := st.chat_input("例如：帮我查一下关于 retinal organoids 的最新文献..."):
    # 显示用户的提问
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Agent 开始接管并处理任务
    with st.chat_message("assistant"):
        # 显示加载动画，此时后台正在疯狂跑 Thought-Action 循环
        with st.spinner('Agent 正在思考并调用科研工具... (可查看终端运行日志)'):
            try:
                # 提取历史对话记录 (简单的字典转换)
                chat_history = []
                for msg in st.session_state.messages[:-1]:
                    chat_history.append({"role": msg["role"], "content": msg["content"]})

                # 见证奇迹的时刻：调用我们刚才写好的最强大脑！
                final_answer = st.session_state.agent.execute(query=prompt, chat_history=chat_history)

                # 在网页上展示最终结果
                st.markdown(final_answer)
                # 将回答存入记忆
                st.session_state.messages.append({"role": "assistant", "content": final_answer})

            except Exception as e:
                error_msg = f"Agent 运行过程中出现异常: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})