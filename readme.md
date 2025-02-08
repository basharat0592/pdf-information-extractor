# 📄 PDF Information Extractor with CrewAI

This is a **Streamlit-based** application that extracts **specific info like premium, tax, and fee details** from PDF files using **CrewAI, LangChain, and FAISS**. It leverages **Ollama's DeepSeek LLM** to accurately extract financial details.

## 🚀 Features
- Upload a **PDF** and extract premium, tax, and fee details.
- Uses **FAISS vector search** for accurate information retrieval.
- Ensures **precise extractions** with page references.
- **Fast & Local Processing** (No API keys required).
- **Persists data in session state** to avoid redundant processing.

## 🛠 Installation
1. **Clone the repository**  
   ```sh
   git clone https://github.com/basharat0592/pdf-information-extractor.git
   cd pdf-information-extractor
