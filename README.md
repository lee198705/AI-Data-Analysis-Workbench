# 小鱼实验室：AI 数据分析工作台

这是一个基于 Streamlit 的数据分析、机器学习建模与智能问答平台。项目当前主线包含：

- 数据工作区：在侧边栏上传或选择一次数据，全站共享当前数据集。
- 数据探索：查看字段、维度、统计摘要和基础图表。
- 机器学习建模：使用无数据泄漏的 sklearn Pipeline 完成预处理、训练、评估和预测。
- 智能问答系统：支持上传文档，使用本地统计知识库 + BM25 检索，并调用硅基流动 API 生成答案。
- 数据可视化：使用 Pygwalker 进行交互式数据分析。

## 项目结构

```text
.
├── app.py                         # Streamlit 主入口
├── apps/                          # 各功能模块
│   ├── eda.py                     # 数据探索
│   ├── models.py                  # Pipeline 模型训练、评估与预测
│   ├── chatting.py                # 智能问答
│   └── analy.py                   # Pygwalker 可视化与内置图表兜底
├── datasets/                      # 示例数据集
├── picture/                       # 页面与问答图片资源
├── imgs/                          # README 演示素材
├── chain.py                       # 硅基流动 OpenAI 兼容 API 调用链
├── rag/                           # 本地知识库与上传文档的 BM25 检索
├── 统计理论知识语料库.xlsx
├── ciyun.txt                      # 停用词表
└── requirements.txt
```

## 机器学习链路

机器学习模块已经按 P0 正确性要求重构。现在训练流程为：

```text
Raw data
→ train/test split
→ Pipeline(preprocess → feature selection → model)
→ fit(train)
→ evaluate(test)
```

其中：

- 先划分训练集和测试集，再在训练集上拟合预处理步骤，避免数据泄漏。
- 自动区分数值特征和类别特征。
- 数值特征：缺失值中位数填充 + 标准化。
- 类别特征：缺失值众数填充 + 稀疏 One-Hot 编码，并合并低频类别。
- 默认使用全部特征；高级模式下可手动启用 `SelectKBest` 特征选择。
- 支持分层抽样；当类别样本数不足时自动退回普通划分。
- 训练后如果切换数据集、目标字段、任务类型、算法或测试集比例，旧 Pipeline 会自动失效。
- 训练集上展示轻量交叉验证分数，独立测试集只用于最终评估。
- 训练完成后保存完整 Pipeline、标签编码器和特征字段，预测新 CSV 时复用同一套处理链路。
- 输出准确率、精确率、召回率、F1 值、二分类 ROC-AUC、混淆矩阵和分类报告。
- 对类别型目标执行分类任务；对连续数值目标可切换为回归任务，输出 MAE、MSE、RMSE 和 R²。

## 数据工作区

侧边栏提供统一数据工作区。用户可以选择示例数据集，也可以上传 CSV、TXT 或 XLSX 文件。当前数据集会保存到 `st.session_state`，数据探索、机器学习、数据可视化和智能问答都会优先使用这份共享数据。

数据读取层只删除完全空白的行，不会提前删除含有部分缺失值的样本。特征缺失值由机器学习 Pipeline 中的 `SimpleImputer` 处理，目标字段缺失值只在建模页单独删除。

## 智能问答链路

问答模块已经按 P1 重构为轻量 RAG。当前流程为：

```text
统计理论知识语料库.xlsx / 上传文档
→ 文本抽取与切块
→ jieba 分词与停用词过滤
→ BM25 Top-K 检索
→ 将来源片段写入 Prompt
→ 硅基流动模型生成带来源依据的回答
```

其中：

- 支持上传 CSV、XLSX、PDF、TXT、DOCX。
- 本地统计知识库会被缓存，避免每次提问重复读取 Excel。
- 上传表格会保留字段、行列数和前 200 行预览作为检索上下文。
- 数据工作区中的当前数据集也会自动进入问答检索上下文。
- 最近多轮对话会真实传入模型，支持“它/刚才那个问题”这类追问。
- 上传文档使用 SHA256 判断内容变化；清空上传框后会清除旧文档上下文。
- 回答下方会展示本次检索命中的来源片段和分数，方便检查答案依据。
- BM25 检索为项目内纯 Python 实现，P1 阶段不需要额外下载向量模型、FAISS 或 reranker。

## 运行方式

建议使用 Python 3.12。当前依赖版本已在 `requirements.txt` 中固定。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export SILICONFLOW_API_KEY="你的硅基流动 API Key"
# 可选：指定模型，默认 deepseek-ai/DeepSeek-V4-Flash
export SILICONFLOW_MODEL="deepseek-ai/DeepSeek-V4-Flash"
streamlit run app.py
```

Windows PowerShell 可以使用：

```powershell
$env:SILICONFLOW_API_KEY="你的硅基流动 API Key"
$env:SILICONFLOW_MODEL="deepseek-ai/DeepSeek-V4-Flash"
streamlit run app.py
```

也可以在本地创建 `.streamlit/secrets.toml`，部署到 Streamlit Cloud 时同样使用 Secrets 面板配置：

```toml
SILICONFLOW_API_KEY = "你的硅基流动 API Key"
SILICONFLOW_MODEL = "deepseek-ai/DeepSeek-V4-Flash"
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
```

## 上传 GitHub 前的说明

本仓库已经移除了本地大模型、IDE 配置、缓存文件、编译产物、旧版实验目录和独立客户端目录。后续如需使用大型模型文件，请通过下载脚本或 README 说明让使用者自行获取，不建议直接提交到 GitHub。

不要提交以下内容：

- `.venv/`
- `__pycache__/`
- `.DS_Store`
- `.env`
- `.streamlit/secrets.toml`
- 本地模型目录或大体积中间产物
