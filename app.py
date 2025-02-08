import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from crewai import Agent, Task, Crew, Process
from crewai_tools import PDFSearchTool
from crewai import LLM

os.environ["OPENAI_API_KEY"] = "bh-proj-123"
st.set_page_config(page_title="PDF Premium Extractor", page_icon="📄")
st.title("📄 PDF Premium Extractor with CrewAI")

# ✅ Upload PDF
pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])

if pdf_file and "vectorstore" not in st.session_state:
    st.write("✅ PDF uploaded successfully.")
    with st.spinner("Processing the PDF..."):
        with open("temp.pdf", "wb") as f:
            f.write(pdf_file.read())

        # ✅ Load and Process PDF
        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()

        # ✅ Ensure Proper Chunking
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        documents_split = splitter.split_documents(documents)

        # ✅ Use Local Embeddings (No API Key Needed)
        embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        # ✅ Create FAISS Vector Store and Store in Session State
        st.session_state.vectorstore = FAISS.from_documents(documents_split, embedding_model)

        # ✅ Fix: Manually Set Embeddings for PDFSearchTool
        st.session_state.pdf_tool = PDFSearchTool(
            vectorstore=st.session_state.vectorstore,
            embedding_model=embedding_model
        )

        # ✅ Use Ollama (DeepSeek LLM)
        st.session_state.llm = LLM(
            model="ollama/deepseek-r1:1.5b",
            base_url="http://localhost:11434",
            temperature=0
        )

        # ✅ CrewAI Agent
        st.session_state.agent = Agent(
            role="PDF Analyst",
            goal="Find premium, tax, and fee details from the PDF accurately",
            backstory="An expert AI that extracts key financial details from documents without fabricating information.",
            tools=[st.session_state.pdf_tool],
            llm=st.session_state.llm
        )

        # ✅ CrewAI Task
        st.session_state.task = Task(
            description="Extract only the explicitly mentioned premium amount, tax, and fee details from the PDF without assuming values.",
            expected_output="Extracted values from the document with exact references.",
            agent=st.session_state.agent
        )

        # ✅ Crew Setup
        st.session_state.crew = Crew(
            agents=[st.session_state.agent],
            tasks=[st.session_state.task],
            process=Process.sequential
        )
    
    st.success("✅ PDF processing completed. Ready for extraction.")

# ✅ Extract Premium Button
if "crew" in st.session_state and st.button("Extract Premium, Tax & Fees"):
    with st.spinner("Extracting premium, tax, and fee details..."):
        query = "Extract explicitly mentioned premium amount, tax, and fee details as stated in the document. Return only the extracted values with page reference."
        log_messages = ["🔍 Query initiated for extracting premium, tax, and fee details."]
        response = st.session_state.crew.kickoff(inputs={"question": query})

        log_messages.append("✅ Extraction completed successfully.")
    
    st.write("📝 Extracted Information:", response)
    
    with st.expander("🔍 Extraction Logs"):
        for log in log_messages:
            st.write(log)
