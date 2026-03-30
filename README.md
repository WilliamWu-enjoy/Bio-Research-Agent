# 🧬 Bio-Research Agent (生物科研智能体)

基于 LangChain 现代架构与 Streamlit 开发的垂直领域科研辅助大模型
Agent。专为生命科学实验场景（如类器官培养、电生理信号分析等）设计，旨在解决通用大模型在严谨学术场景下的“严重幻觉”与“执行力缺失”痛点。

## 🌟 核心痛点与解决方案

通用大模型（如 GPT-4）在科研辅助中存在知识非实时、缺乏私有 Protocol 数据、以及数值计算极易出错等问题。本项目通过 **RAG (
检索增强生成)** 与 **ReAct (推理与行动)** 双重架构解决上述问题：

- **受控知识边界 (RAG)**：通过挂载本地 PDF 文献库，强制大模型基于真实检索结果输出（如精确的视网膜类器官电刺激方案），根除参数编造幻觉。
- **动态工具调用 (Agent Tools)**：封装高精度 Python 工具栈，将数学计算和权威数据检索交由专有工具执行，模型仅负责意图识别与任务分发。

## 🛠 技术架构与核心特性

- **前端交互**: `Streamlit` 实现带思维链（Thought Process）透明展示的流式问答 UI。
- **底层执行器**: 采用最新版 LangChain 架构 (`create_agent`) 构建 ReAct 调度引擎，配置 `recursion_limit` 防止工具调用死循环，并实现
  Tool-level 参数纠错自愈。
- **知识库管道**: `ChromaDB` 本地向量检索，引入重叠区（Overlap）切分算法保留实验步骤连贯性，并内置 Query Translation
  缓解中英跨语言检索的语义衰减。
- **多源工具箱**:
    - `PubMed Sniffer`: 基于 NCBI Entrez API 的前沿文献实时动态嗅探。
    - `C1V1 Calculator`: 严格的代码级实验室试剂配比/浓度稀释计算器。
    - `MEA Signal Analyzer`: 模拟微电极阵列 (MEA) 电生理特征提取流程（扩展预留）。

## 🚀 快速启动

### 1. 克隆项目与环境配置

```bash
git clone [https://github.com/YourUsername/Bio-Research-Agent.git](https://github.com/YourUsername/Bio-Research-Agent.git)
cd Bio-Research-Agent

# 推荐使用虚拟环境
conda create -n bio_agent python=3.10
conda activate bio_agent

# 安装依赖
pip install -r requirements.txt
```

### 2. 环境配置变量
#### 复制根目录下的 .env.example 并重命名为 .env，填入你的大模型 API 密钥：
```bash
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxx
OPENAI_BASE_URL=[https://api.your-proxy.com/v1](https://api.your-proxy.com/v1)
```

### 3.构建本地知识库（首次运行）
#### 将你的目标科研文献 (PDF 格式) 放入 data/ 目录中，然后执行向量化脚本：
```bash
python vector_store.py
```

### 4.启动Web服务
```bash
streamlit run app.py
```

## 项目结构
```bash
Bio-Research-Agent/
│
├── data/                      # 存放原始数据的目录 (建议不要传太多几十兆的 PDF)
│   └── ncomms5047.pdf         # 你的测试文献
│
├── chroma_db/                 # 向量数据库的持久化目录 (必须加入 .gitignore)
│
├── app.py                     # [前端入口] Streamlit 网页交互应用
├── react_agent.py             # [核心调度] 基于 LangChain 最新 API 的 ReAct 智能体大脑
├── agent_tools.py             # [工具箱] 包含 PubMed联网、试剂计算器等自定义 Tools
├── vector_store.py            # [数据管道] PDF 文档加载、切分与 Chroma 向量化逻辑
├── rag_service.py             # [知识检索] RAG 检索器与 Prompt 翻译拼装逻辑
├── factory.py                 # [模型工厂] LLM 模型统一初始化与切换配置
│
├── requirements.txt           # [项目依赖] 记录所需的所有 Python 包及版本
├── .env.example               # [环境变量模板] 告诉其他人需要配置哪些 API Key
├── .gitignore                 # [Git 忽略清单] 防止私钥、大文件和虚拟环境被传到网上
└── README.md                  # [项目说明书] 整个仓库的门面！
```

## 📝 TODO / 未来迭代计划
- **接入 Playwright 自动化浏览器，实现特定商业文献站点的免密自动化抓取。**
- **升级底层的路由分发，向 Multi-Agent (多智能体协作) 架构演进。**