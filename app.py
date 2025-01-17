import os
import pandas as pd
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import gradio as gr

# Çevresel değişkenleri yükleme
load_dotenv()
embedding_model_path = "sentence-transformers/all-MiniLM-L6-v2"

# Belgeleri yükleme
def load_documents():
    loaders = [
        PyPDFLoader("docs/Grup_TeminatTablosu1.pdf"),
        PyPDFLoader("docs/Grup_TeminatTablosu2.pdf"),
        PyPDFLoader("docs/Kisiye_Ozel_Saglik_Sigortasi_Ozel_Sartlari.pdf"),
        PyPDFLoader("docs/Kisiye_Ozel_Saglik_Sigortasi_Teminat_Tablosu1.pdf"),
        PyPDFLoader("docs/Kisiye_Ozel_Saglik_Sigortasi_Teminat_Tablosu2.pdf"),
        PyPDFLoader("docs/Kisiye_Ozel_TSS_Ozel_Sartlari.pdf"),
        PyPDFLoader("docs/Kisiye_Ozel_TSS_Teminat_Tablosu1.pdf"),
        PyPDFLoader("docs/Kisiye_Ozel_TSS_Teminat_Tablosu2.pdf"),
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

        # Document formatında oluştur
        documents.append(Document(page_content=f"Soru: {question}\nCevap: {combined_answer}", metadata={"source": "Excel"}))

    return documents

# Belgeleri parçalara ayırma ve vektör veritabanı oluşturma
def create_vector_store(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents)

    # Embedding işlemini uygula ve FAISS oluştur
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_path)
    vector_store = FAISS.from_documents(split_docs, embeddings)
    return vector_store

# Sorgulama motoru
def get_answer(question, vector_store):
    retriever = vector_store.as_retriever()
    result = retriever.get_relevant_documents(question)
    combined_answer = "\n\n".join([doc.page_content for doc in result])
    return combined_answer

# Ana UI
def chatbot_ui():
    documents = load_documents()
    vector_store = create_vector_store(documents)

    def chatbot_response(question):
        return get_answer(question, vector_store)

    iface = gr.Interface(fn=chatbot_response, inputs="text", outputs="text", title="Sigorta Bilgi Botu")
    iface.launch()

if __name__ == "__main__":
    chatbot_ui()
