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

# Belge grupları
priority_documents = {
    "special_conditions": [
        "docs/Grup_Policesi_Ozel_Sartlari.docx",
        "docs/Kisiye_Ozel_Saglik_Sigortasi_Ozel_Sartlari.pdf",
        "docs/Kisiye_Ozel_TSS_Ozel_Sartlari.pdf"
    ],
    "coverage_tables": [
        "docs/Grup_TeminatTablosu1.pdf",
        "docs/Grup_TeminatTablosu2.pdf",
        "docs/Kisiye_Ozel_Saglik_Sigortasi_Teminat_Tablosu1.pdf",
        "docs/Kisiye_Ozel_Saglik_Sigortasi_Teminat_Tablosu2.pdf",
        "docs/Kisiye_Ozel_TSS_Teminat_Tablosu1.pdf",
        "docs/Kisiye_Ozel_TSS_Teminat_Tablosu2.pdf"
    ]
}

# Belgeleri yükleme
def load_documents():
    documents = []

    for doc_type, file_paths in priority_documents.items():
        for path in file_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")

            if path.endswith(".pdf"):
                loader = PyPDFLoader(path)
            elif path.endswith(".docx"):
                loader = UnstructuredWordDocumentLoader(path)
            else:
                continue

            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata["type"] = doc_type  # Doküman türünü metadata olarak ekle
                documents.append(doc)

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

# Özel sıralama mantığı
class PriorityRetriever:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def retrieve(self, query):
        all_results = self.vector_store.similarity_search(query, k=10)
        # Özel şartlar öncelikli, ardından teminat tabloları
        special_results = [doc for doc in all_results if doc.metadata.get("type") == "special_conditions"]
        coverage_results = [doc for doc in all_results if doc.metadata.get("type") == "coverage_tables"]

        # Öncelikli sonuçları birleştir
        ordered_results = special_results + coverage_results
        return ordered_results

# Soruya yanıt verme
def get_answer(question, vector_store):
    retrieverprint = vector_store.as_retriever()
    docs = retrieverprint.get_relevant_documents(question)
    print("Retrieved Documents:", docs)
    
    retriever = PriorityRetriever(vector_store)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
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
