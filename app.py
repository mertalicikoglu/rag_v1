import os
from langchain.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.schema import Document
from dotenv import load_dotenv
import gradio as gr
import pandas as pd
import nltk
import openai

# Gerekli NLTK kaynaklarını indir
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')

# OpenAI API Anahtarını yükle
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found. Please set it in your .env file.")


openai.api_key = openai_api_key

# Belgeleri yükleme
def load_documents():
    file_paths = [
        "docs/Grup_TeminatTablosu1.pdf",
        "docs/Grup_TeminatTablosu2.pdf",
        "docs/Kisiye_Ozel_Saglik_Sigortasi_Ozel_Sartlari.pdf",
        "docs/Kisiye_Ozel_Saglik_Sigortasi_Teminat_Tablosu1.pdf",
        "docs/Kisiye_Ozel_Saglik_Sigortasi_Teminat_Tablosu2.pdf",
        "docs/Kisiye_Ozel_TSS_Ozel_Sartlari.pdf",
        "docs/Kisiye_Ozel_TSS_Teminat_Tablosu1.pdf",
        "docs/Kisiye_Ozel_TSS_Teminat_Tablosu2.pdf",
        "docs/Grup_Policesi_Ozel_Sartlari.docx"
    ]

    for path in file_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

    loaders = [
        PyPDFLoader(file_paths[0]),
        PyPDFLoader(file_paths[1]),
        PyPDFLoader(file_paths[2]),
        PyPDFLoader(file_paths[3]),
        PyPDFLoader(file_paths[4]),
        PyPDFLoader(file_paths[5]),
        PyPDFLoader(file_paths[6]),
        PyPDFLoader(file_paths[7]),
        UnstructuredWordDocumentLoader(file_paths[8])
    ]
    
    documents = []
    for loader in loaders:
        documents.extend(loader.load())

    # Excel dosyasını yükleme
    excel_path = "docs/Chatbot_Demo_Sorular.xlsx"
    df = pd.read_excel(excel_path)

    for _, row in df.iterrows():
        question = str(row.get("Soru", "")).strip()
        answer_part1 = str(row.get("Standart Madde 1", "")).strip()
        answer_part2 = str(row.get("Standart Madde 2", "")).strip()
        combined_answer = f"{answer_part1}\n\n{answer_part2}"
        
        documents.append(Document(page_content=f"Soru: {question}\nCevap: {combined_answer}", metadata={"source": "Excel"}))

    return documents

# Belgeleri vektör veritabanına dönüştürme
def create_vector_store(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents)
    embedding_model_path = "sentence-transformers/paraphrase-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_path)
    vector_store = FAISS.from_documents(split_docs, embeddings)
    return vector_store

# Soruya yanıt verme
def get_answer(question, vector_store):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vector_store.as_retriever())
    return qa_chain.run(question)

# Gradio arayüzü
def chatbot_ui():
    documents = load_documents()
    vector_store = create_vector_store(documents)

    def chatbot_response(question):
        return get_answer(question, vector_store)

    iface = gr.Interface(fn=chatbot_response, inputs="text", outputs="text", title="Sigorta Bilgi Botu")
    iface.launch()

if __name__ == "__main__":
    chatbot_ui()
