## 项目实现流程

1. 构建项目的 log 类，使用 loguru 库
2. 整个项目分为两个部分，第一部分为 mysql qa，第二部分为 rag qa
   1. mysql qa 主要使用 mysql redis bm25
   2. rag qa 主要使用 langchain 搭建 rag workflow

## workflow
第一步：将现有后台搜集的FQA数据集存储到Mysql数据库中

第二步：基于query实现Mysql数据库检索：将query和现有问题匹配（做相似度计算），如果阈值>=0.85，就认为问题比较明确，直接返回对应的答案；否则，进入RAG检索系统

第三步：搭建本地知识库：对本地文档加载读取；进行文档分割；文档向量化；存储向量数据库（Milvus）

第四步：基于query实现Milvus数据库检索：将query进行向量表示，并从Milvus数据库中检索出相似的top-k个文本段。

第五步：将query和检索出的top-k文本段拼接，送入大模型，实现预测。

## 项目结构

- `base`：
  - `config.py`：配置管理，加载 yaml 文件
  - `logger.py`：日志设置，使用 loguru
- `core`
  - `rag`
    - `prompt.py`：rag 和 llms 的提示词
    - `query.py`： 查询分类器
    - `retrieval.py`：检索选择器
    - `vector.py`：向量存储和检索
    - `system.py`：rag 系统的核心逻辑
  - `sql`
    - `cache.py`：Redis 缓存操作
    - `db.py`：mysql 数据库操作
    - `retrieval.py`：bm25 检索
    - `utils.py`：文本预处理
- `main.py`：集成系统主入口

大模型通过 Ollama 部署本地