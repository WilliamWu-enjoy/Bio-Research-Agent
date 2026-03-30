from factory import get_chat_model
from vector_store import VectorStoreService
from langchain_core.prompts import ChatPromptTemplate


class RagSummarizeService:
    def __init__(self):
        """
        初始化 RAG 服务 (组装流水线)
        对应架构图中的 __init__ 方法
        """
        # 1. 引入检索器 (超级图书管理员)
        self.vs_service = VectorStoreService()
        self.retriever = self.vs_service.get_retriever(search_kwargs={"k": 3})

        # 2. 引入大模型 (聪明的实习生)
        self.model = get_chat_model()

        # 3. 加载提示词模板 (印制考卷规则)
        self.prompt_template = self._load_prompt_text()

    def _load_prompt_text(self):
        """
        对应图中的 def _load_prompt_text
        这里是核心！通过 Prompt Engineering 控制大模型必须基于文献用中文回答。
        """
        system_prompt = """你是一个严谨的生物医学科研助手。
        请严格根据以下提供的文献片段（Context）来回答用户的问题。

        【重要规则】：
        1. 如果文献片段是英文，请你充分理解后，用专业、流畅的【中文】进行总结和回答。
        2. 如果提供的文献片段中没有能回答该问题的信息，请直接回答：“抱歉，在本地文献库中未能找到与该问题相关的精确描述。” 绝不能凭借你自己的记忆编造数据。

        已知文献片段：
        {context}
        """

        # 组装成 LangChain 的 Prompt 模板格式
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}"),
        ])

    def retrieve_docs(self, query):
        """
        对应图中的 def retrieve_docs
        拿着问题去档案室找小抄
        """
        return self.retriever.invoke(query)

    def rag_summarize(self, query):
        """
        对应图中的 def rag_summarize
        这是对外暴露的主干方法，将被封装成 Agent 的 Tool
        """
        # 1. 找文献小抄
        docs = self.retrieve_docs(query)

        if not docs:
            return "抱歉，文献库中没有找到相关内容。"

        # 2. 把找出来的 Document 对象，提取出里面的文字，拼成一个长字符串
        # 这就是真正要塞给大模型的 {context}
        context_text = "\n\n".join([doc.page_content for doc in docs])   #这行代码在做什么？

        # 3. 把“小抄”和“用户的问题”填进模板里
        messages = self.prompt_template.invoke({
            "context": context_text,
            "question": query
        })

        # 4. 把填好的考卷交给大模型生成最终的中文答案
        response = self.model.invoke(messages)

        # 返回大模型生成的字符串文本
        return response.content


# ==========================================
# 本地测试代码 (看看魔法是否生效)
# ==========================================
if __name__ == "__main__":
    # 实例化 RAG 服务
    rag_service = RagSummarizeService()

    # 既然你的本地文献是关于 3D optic cup (视网膜类器官) 的，我们问一个更有针对性的问题
    query = "请详细说明在培养 3D optic cup (视网膜类器官) 时，文献中提到了哪些关键的发展阶段或现象？"

    print(f"\n[用户提问]: {query}")
    print("\n[RAG 系统正在思考并生成答案...]\n")

    # 调用核心方法
    final_answer = rag_service.rag_summarize(query)

    print("-" * 50)
    print(final_answer)
    print("-" * 50)