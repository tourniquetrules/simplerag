# 🚑 Emergency Medicine RAG Chat Interface

Advanced RAG system for emergency medicine using medical literature and abstracts.

## 🚀 Quick Setup

### Prerequisites
- Python 3.11 or higher
- LM Studio running with required models

### 📦 Installation

1. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run the application:**
```bash
python3 emergency_rag_chatbot.py
```

3. **Access the interface:**
   - Open your browser to: `http://localhost:7866`

### 🔧 Configuration

**Important:** Update the API endpoint in `emergency_rag_chatbot.py` to match your setup:

```python
# Change this to your local computer, OpenAI, or other endpoint
LM_STUDIO_BASE_URL = "http://10.5.0.2:1234"
```

### 🤖 Required Models

This application assumes you are using the following models in LM Studio:

#### LLM Model
```
deepseek/deepseek-r1-0528-qwen3-8b
```

#### Embedding Model  
```
text-embedding-all-minilm-l6-v2-embedding
```

**Note:** If you're using different models, update the model names in the configuration section of `emergency_rag_chatbot.py`.

### 📚 Usage

1. **Upload PDFs:** Use the Knowledge Base Management section to upload emergency medicine abstracts
2. **Ask Questions:** Type your medical questions in the chat interface
3. **General Knowledge:** Use `@llm` prefix for non-medical questions that bypass the RAG system

### 🛠️ Advanced Setup

<details>
<summary>Alternative Installation Methods</summary>

#### Using virtual environment (recommended):
```bash
python3 -m venv emarag-env
source emarag-env/bin/activate  # On Windows: emarag-env\Scripts\activate
pip install -r requirements.txt
```

#### Manual dependency installation:
```bash
pip install gradio requests PyPDF2 numpy faiss-cpu
```

</details>

### 📋 System Requirements

- **Python:** 3.11+
- **RAM:** 4GB+ recommended
- **Storage:** 1GB+ for models and documents
- **Network:** Access to LM Studio API endpoint

---

**Model Configuration:**
- **LLM:** `deepseek/deepseek-r1-0528-qwen3-8b`
- **Embedding:** `text-embedding-all-minilm-l6-v2-embedding`
- **Focus:** Emergency Medicine Research & Practice

💡 **Tip:** Use `@llm` at the start of your question for general knowledge (bypasses RAG)

