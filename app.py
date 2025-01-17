import os
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import gradio as gr
import pandas as pd

# Yerel embedding modeli
class LocalEmbedding:
    def __init__(self):
        # `all-MiniLM-L6-v2` modeli hızlı ve hafif bir embedding modelidir.
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_tensor=False).tolist()

# Embedding modeli örneği
embedding = LocalEmbedding()

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

        documents.append(Document(page_content=f"Soru: {question}\nCevap: {combined_answer}", metadata={"source": "Excel"}))

    return documents

# Belgeleri parçalara ayırma ve vektör veritabanı oluşturma
def create_vector_store(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents)
    texts = [doc.page_content for doc in split_docs]
    metadatas = [doc.metadata for doc in split_docs]

    # FAISS vektör veritabanı oluşturma
    vector_store = FAISS.from_texts(texts, embedding.model.encode, metadatas=metadatas)
    return vector_store

# Sorgulama motoru
def get_answer(question, vector_store):
    # Soru için embedding oluştur ve en yakın belgeleri getir
    query_embedding = embedding.model.encode([question])[0]
    results = vector_store.similarity_search_by_vector(query_embedding, k=5)

    # Yanıtları birleştir
    combined_answer = "\n\n".join([result.page_content for result in results])
    return combined_answer

# Gradio UI
def chatbot_ui():
    documents = load_documents()
    vector_store = create_vector_store(documents)

    def chatbot_response(question):
        return get_answer(question, vector_store)

    iface = gr.Interface(fn=chatbot_response, inputs="text", outputs="text", title="Sigorta Bilgi Botu")
    iface.launch()

if __name__ == "__main__":
    chatbot_ui()
