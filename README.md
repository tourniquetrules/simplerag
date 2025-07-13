# 🚑 Emergency Medicine RAG Chat Interface

Advanced RAG system for emergency medicine using medical literature and abstracts with **Clinical-BERT** medical embeddings.

## 🏥 Key Features

- **Medical-Specialized Embeddings:** Clinical-BERT for superior medical text understanding
- **Hybrid Fallback System:** LM Studio API backup for reliability  
- **Emergency Medicine Focus:** Optimized for emergency protocols and evidence-based care
- **Session Isolation:** Clean separation between RAG and general knowledge queries
- **Cloudflare Tunnel Support:** Secure remote access via `emarag.haydd.com`

## 🚀 Quick Setup

### Prerequisites
- Python 3.11 or higher
- LM Studio running with required models
- CUDA-compatible GPU (recommended for Clinical-BERT)

### 📦 Installation

1. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

If you don't have a requirements.txt, install manually:
```bash
pip install gradio requests PyPDF2 numpy faiss-cpu sentence-transformers transformers torch
```

2. **Run the application:**
```bash
python3 emergency_rag_chatbot.py
```

3. **Access the interface:**
   - **Local:** `http://localhost:7866`
   - **Remote:** `https://emarag.haydd.com` (if Cloudflare tunnel configured)

### 🔧 Configuration

**Important:** Update the API endpoint in `emergency_rag_chatbot.py` to match your setup:

```python
# Change this to your local computer, OpenAI, or other endpoint
LM_STUDIO_BASE_URL = "http://10.5.0.2:1234"
```

### 🤖 Model Configuration

#### 🧠 LM Studio Setup
Ensure LM Studio is running on port 1234 with:
- **Chat Model:** `deepseek/deepseek-r1-0528-qwen3-8b`
- **Local Server:** `http://localhost:1234/v1`

#### 🔬 Clinical-BERT Embeddings  
The system uses **Clinical-BERT** (`emilyalsentzer/Bio_ClinicalBERT`) for medical-specialized embeddings:

- **Primary:** Clinical-BERT for superior medical text understanding
- **Fallback:** LM Studio text embeddings API for reliability
- **GPU Support:** Automatic CUDA utilization (~500MB-1GB VRAM)
- **Auto-Loading:** Downloads on first run (~500MB), cached locally

#### 🔄 Hybrid Embedding System
```
Medical Text → Clinical-BERT (local GPU) → FAISS Index
             ↓ (if Clinical-BERT unavailable)
             → LM Studio API → FAISS Index  
```

**Benefits of Clinical-BERT:**
- Medical terminology understanding
- Clinical concept relationships
- Evidence-based medicine optimization
- Superior performance on medical literature

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
- **RAM:** 4GB+ recommended (8GB+ for optimal performance)
- **Storage:** 2GB+ free space (Clinical-BERT model cache)
- **GPU:** CUDA-compatible GPU recommended for Clinical-BERT
  - **VRAM:** 500MB-1GB for Clinical-BERT embeddings
  - **Fallback:** CPU processing available if no GPU
- **Network:** Internet connection for initial Clinical-BERT download

### 🔧 Dependencies

#### Core Libraries
```bash
pip install gradio requests PyPDF2 numpy faiss-cpu
```

#### Medical Embedding Support  
```bash
pip install sentence-transformers transformers torch
```

#### GPU Support (optional but recommended)
```bash
# For NVIDIA GPUs
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### ⚡ Performance Notes

- **First Run:** Clinical-BERT downloads automatically (~500MB)
- **GPU Usage:** Clinical-BERT utilizes CUDA for faster embeddings
- **Fallback Mode:** Automatically switches to LM Studio if Clinical-BERT unavailable
- **Memory:** Clinical-BERT loads into GPU memory for optimal performance
- **Storage:** 1GB+ for models and documents
- **Network:** Access to LM Studio API endpoint

---

**Model Configuration:**
- **LLM:** `deepseek/deepseek-r1-0528-qwen3-8b`
- **Embedding:** `text-embedding-all-minilm-l6-v2-embedding`
- **Focus:** Emergency Medicine Research & Practice

💡 **Tip:** Use `@llm` at the start of your question for general knowledge (bypasses RAG)
