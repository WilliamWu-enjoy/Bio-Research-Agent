import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

# 加载 .env 里的 API Key 和 Base URL
load_dotenv()


class VectorStoreService:
    def __init__(self, persist_directory="./chroma_db"):
        """
        初始化向量数据库服务
        persist_directory: 数据库在本地保存的文件夹路径
        """
        self.persist_directory = persist_directory

        # 1. 初始化 Embedding 模型 (负责把文字变成数学向量)
        # 这里复用你在 .env 里配置的中转站 Key，大部分中转站都支持 text-embedding-3-small
        self.embeddings = OpenAIEmbeddings(
            model="Qwen/Qwen3-Embedding-8B",
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL")
        )

        # 2. 初始化切分器 (对应你图里的 self.spliter)
        # 面试必考：chunk_size 是每块的大小，chunk_overlap 是块与块之间重叠的字数，防止上下文断裂
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""]
        )

        # 3. 准备用来装数据库的容器
        self.vector_store = None

    def load_document(self, file_path):
        """
        对应你图里的 def load_document
        读取 PDF，切分，并存入向量数据库
        """
        print(f"正在读取并解析文献: {file_path} ...")
        # 读取 PDF
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        # 切分文档
        split_docs = self.spliter.split_documents(docs)
        print(f"文献已被切分成 {len(split_docs)} 个文本块。正在进行向量化并存入数据库...")

        # 将切分好的文档存入 Chroma 数据库，并持久化到本地
        self.vector_store = Chroma.from_documents(
            documents=split_docs,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        print("本地文献库构建完成！")

    def get_retriever(self, search_kwargs={"k": 3}):
        """
        对应你图里的 def get_retriever
        返回一个检索器，供 Agent 或 RAG 服务调用
        search_kwargs: {"k": 3} 表示每次搜索返回最相关的前 3 个段落
        """
        # 如果内存里没有，尝试从本地硬盘加载已经建好的数据库
        if self.vector_store is None:
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )

        return self.vector_store.as_retriever(search_kwargs=search_kwargs)


# ==========================================
# 本地测试代码 (只有直接运行这个文件时才会执行)
# ==========================================
if __name__ == "__main__":
    # 假设你在项目里建了一个 data 文件夹，里面放了一篇叫 protocol.pdf 的文献
    # 你可以先随便找个 PDF 改名叫 protocol.pdf 放进去测试
    test_pdf_path = "./data/ncomms5047.pdf"

    # 实例化服务
    vs_service = VectorStoreService()

    # 如果本地有 PDF，我们跑一下加载和切分
    if os.path.exists(test_pdf_path):
        vs_service.load_document(test_pdf_path)

        # 测试一下检索器是不是真的好用
        retriever = vs_service.get_retriever(search_kwargs={"k": 2})
        query = "请问培养过程中需要注意哪些关键参数？"  # 根据你上传的 PDF 内容随便问一个问题

        print(f"\n正在搜索问题: {query}")
        results = retriever.invoke(query)

        print(f"\n找到了 {len(results)} 段最相关的内容：")
        for i, res in enumerate(results):
            print(f"--- 结果 {i + 1} ---")
            # 打印找出来的文本和它所在的页码
            print(f"来源页码: {res.metadata.get('page', '未知')}")
            print(f"内容片段: {res.page_content[:150]}...\n")
    else:
        print(f"找不到测试文件：{test_pdf_path}，请先准备一个 PDF 文献用于测试。")