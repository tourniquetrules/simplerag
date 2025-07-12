# simplerag
This is a simple chatbot with the addition of RAG capability for PDFs.  It is focused on emergency medicine.
I used LM Studio API running on the network for the backend, running in Windows so that I could see the different parameters that are available.  Ollama works too but I found LM Studio easier.
I used Python 3.11 as Python 3.12 had too many problems with PDF related dependencies.  It could be me...
I used VS Code Copilot with Claude Sonnet 4 to generate code.
I am using Gradio to generate the webpage and Cloudflare tunnels to host it from my PC.
In general to get this running, 

download LM Studio from https://lmstudio.ai/
Install python 3.11 and pip
then pip install -r requirements.txt
Open emergency_rag_chatbot.py and change your endpoint to your local computer or OPENAI or whatever you pick.  Beyond this it "might" work.
then python3 emergency_rag_chatbot.py

The general flow through the chatbot is:

PDFs → PyPDF2 → chunk_text() → LM Studio Embedding API → FAISS Index
                                                              ↓
User Query → LM Studio Embedding API → FAISS Search → Top Chunks
                                                              ↓
Chunks + Query → create_rag_prompt() → DeepSeek LLM → Final Answer

All abstracts or PDFs are in a subdirectory called abstracts.  

🔧 Chunking
Software: Custom Python function chunk_text()

Framework: Pure Python with regex
Method: Text splitting with sentence boundary detection
Size: 512 characters per chunk with 50-character overlap
Logic: Tries to break at sentence endings (.!?) within 100 characters of the target boundary

🔗 Embedding
Software: LM Studio API

Model: text-embedding-all-minilm-l6-v2-embedding
Framework: HTTP API calls to your LM Studio server
Function: call_embedding_api() sends text to /v1/embeddings endpoint
Output: Vector embeddings (numerical representations of text meaning)

)
🗄️ Database/Storage
Vector Database: FAISS (Facebook AI Similarity Search)

Type: In-memory vector database
Index: faiss.IndexFlatIP (Inner Product for cosine similarity)
Storage: document_store[] (Python list) for metadata + text
Persistence: RAM only (lost on restart unless you add persistence)

🔍 Similarity Search & Retrieval
Software: FAISS + Custom logic

Process:
Query Embedding: User question → LM Studio embedding API
Vector Search: FAISS finds most similar chunks
Filtering: Only chunks above similarity threshold (0.3)
Ranking: Top 5 most relevant chunks returned
Function: retrieve_relevant_chunks()

🤖 Final Answer Generation
Software: LM Studio API

Model: deepseek/deepseek-r1-0528-qwen3-8b
Process:
Context Assembly: create_rag_prompt() combines retrieved chunks with user question
LLM Call: Enhanced prompt sent to DeepSeek model
Response Processing: process_deepseek_response() extracts final answer from reasoning
