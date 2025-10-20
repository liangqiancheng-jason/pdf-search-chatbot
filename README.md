# 多 PDF DeepSeek 聊天代理 🤖

一个基于 Streamlit 的网页应用，通过检索增强生成（RAG）流程实现与多个 PDF 文档的对话。上传 PDF，使用 FAISS 建立向量索引，再利用 DeepSeek 的 `deepseek-chat` 模型和本地 Hugging Face 向量生成器获取基于文档的回答。


---

## 功能亮点
- **多文档问答**：一次会话内即可针对所有已上传的 PDF 提问。
- **本地向量化**：默认使用 Hugging Face 的 `sentence-transformers/all-MiniLM-L6-v2`，无需额外 API 调用即可快速生成向量。
- **可复用的 FAISS 索引**：向量数据库持久化到本地，每次处理后可直接复用。
- **友好的 Streamlit 界面**：侧边栏完成上传与索引，主区域实时展示问答结果。
- **高度可配置**：通过环境变量即可切换嵌入模型或 DeepSeek API 的 Base URL。
- **对话摘要记忆**：自动总结历史对话并注入提示词，回答更连贯、语气更温和。

---

## 工作原理
1. **PDF 文本抽取**：使用 `PyPDF2` 读取每个 PDF 的所有页面文本。
2. **文本分块**：`RecursiveCharacterTextSplitter` 将全文拆成带重叠的片段，覆盖更多上下文。
3. **嵌入生成**：Hugging Face 模型把每个片段编码成密集向量。
4. **向量索引**：FAISS 在本地构建向量数据库，支持高效相似度检索。
5. **DeepSeek 问答**：检索到的片段输入 LangChain QA 链，由 `deepseek-chat` 生成依据文档的答案。


---

## 环境要求
- Python 3.10+
- DeepSeek API Key
- 网络可下载 Hugging Face 向量模型（首次运行需下载权重）

安装依赖：

```bash
pip install -r requirements.txt
```

---

## 环境变量
在项目根目录创建 `.env`（或直接导出环境变量）：

```
DEEPSEEK_API_KEY=<你的-deepseek-api-key>
```

可选配置：
- `DEEPSEEK_API_BASE`（默认 `https://api.deepseek.com/v1`）
- `DEEPSEEK_EMBEDDING_MODEL`（默认 `deepseek-text-embedding`）
- `EMBEDDING_PROVIDER`（默认 `huggingface`，设置为 `deepseek` 时使用云端嵌入）
- `EMBEDDING_MODEL_NAME`（当使用 Hugging Face 时默认 `sentence-transformers/all-MiniLM-L6-v2`）

---

## 启动应用

```bash
streamlit run chatapp.py
```

Streamlit 启动后会自动在浏览器打开页面，默认地址为 `http://localhost:8501`。

---

## 使用指南
- 在侧边栏上传一个或多个 PDF，并点击 **Submit & Process**。
- 等待“Processing…”提示结束，FAISS 索引会以 `faiss_index` 存储在本地。
- 在输入框中使用自然语言提问，答案会引用已建立索引的文档内容。
- 想替换文档时，重新上传并再次点击 **Submit & Process** 即可。

---

## 自定义建议
- 在 `get_text_chunks` 中调整分块大小与重叠长度，以适配不同文档。
- 通过 `EMBEDDING_PROVIDER` 切换 Hugging Face 本地模型或 DeepSeek 云端嵌入；若选择 `huggingface` 可再设置 `EMBEDDING_MODEL_NAME`。
- 若自建或代理 DeepSeek API，可通过 `DEEPSEEK_API_BASE` 指向新的网关。

---

## 常见问题
- **“未检测到 DEEPSEEK_API_KEY”**：启动前确认 `.env` 或环境变量已配置密钥。
- **首次运行较慢**：Hugging Face 向量权重需首次下载，之后会缓存。
- **FAISS 反序列化警告**：加载索引时启用了 `allow_dangerous_deserialization=True`，请只加载可信索引文件。
- **DeepSeek 嵌入 404**：如遇 `openai.NotFoundError`，说明账号暂未开放对应嵌入模型，可改用默认的 `EMBEDDING_PROVIDER=huggingface`。

---