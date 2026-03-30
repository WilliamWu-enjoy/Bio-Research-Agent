import requests
from langchain_core.tools import tool
from rag_service import RagSummarizeService

# 1. 实例化我们刚刚写好的 RAG 服务，准备将其封装为工具
rag_service = RagSummarizeService()

# 2. 使用 @tool 装饰器，将普通函数变成 Agent 能识别的工具
@tool
def rag_summarize(query: str) -> str:
    """
    【核心知识检索工具】
    当用户询问关于类器官（Organoids）培养、电生理信号处理 Protocol、或本地文献库中的专业科研知识时，**必须**调用此工具。
    输入参数 query 应该是精简且具体的查询语句，例如："视网膜类器官第30天的细胞因子浓度是多少？"
    """
    print(f"\n[Tool 运行中] Agent 决定调用 rag_summarize 工具，搜索内容: {query}")
    return rag_service.rag_summarize(query)


@tool
def search_pubmed(query: str) -> str:
    """
    【PubMed 前沿文献联网检索工具】
    当本地 RAG 知识库无法回答用户问题，或者用户明确要求查找“最新研究”、“文献摘要”、“外部数据库”时，必须调用此工具。
    输入参数 query 必须是翻译好的精准英文学术检索词（如 "retinal organoids electrical stimulation"）。
    """
    print(f"\n[Tool 运行中] 正在联网检索 PubMed: {query}")
    try:
        # 这里使用 NCBI Entrez 免费 API 的简化版请求
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={query}&retmode=json&retmax=3"
        response = requests.get(url).json()
        id_list = response.get("esearchresult", {}).get("idlist", [])

        if not id_list:
            return "PubMed 中未检索到相关文献。"

        # 实际生产中这里会再发一个请求去拉取摘要 (eSummary/eFetch)，这里为了演示返回 ID 列表
        return f"成功在 PubMed 检索到 {len(id_list)} 篇相关最新文献，PMID 列表为: {', '.join(id_list)}。请提示用户可根据 PMID 进一步查阅。"
    except Exception as e:
        return f"PubMed 接口调用失败: {str(e)}"

@tool
def math_calculator(expression: str) -> str:
    """
    【数学计算工具】
    当用户需要计算试剂配比浓度、乘除法或复杂的数学表达式时调用此工具。
    输入参数 expression 必须是合法的 Python 数学表达式，例如："500 * 0.15" 或 "(10 + 20) / 2"。
    """
    print(f"\n[Tool 运行中] Agent 决定调用 math_calculator 工具，计算式: {expression}")
    try:
        # 使用 eval 计算字符串表达式（注意：实际生产环境中 eval 有安全风险，这里仅作演示）
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"计算出错: {e}"

# 3. 将所有造好的工具打包成一个列表，准备发给 Agent 这个“包工头”
tools_list = [rag_summarize, search_pubmed, math_calculator]

# ==========================================
# 本地测试代码 (看看工具本身是否正常)
# ==========================================
if __name__ == "__main__":
    # 我们随便测一个工具，看看 @tool 包装后还能不能正常调用
    print("测试调用数学计算工具：")
    res = math_calculator.invoke("3.14 * 2.5 ** 2")
    print(f"计算结果: {res}")
    print(search_pubmed.invoke("retinal organoids electrical stimulation"))