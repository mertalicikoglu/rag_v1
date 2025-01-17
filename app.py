import os
from langchain.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.schema import Document, BaseRetriever
from pydantic import BaseModel
from dotenv import load_dotenv
import gradio as gr
import pandas as pd
import nltk

# Gerekli NLTK kaynaklarını indir
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')

# OpenAI API Anahtarını yükle
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found. Please set it in your .env file.")

# Belgeleri yükleme ve gruplandırma
def load_and_group_documents():
    # Özel Şartlar Dosyaları
    special_files = [
        "docs/Grup_Policesi_Ozel_Sartlari.docx",
        "docs/Kisiye_Ozel_Saglik_Sigortasi_Ozel_Sartlari.pdf",
        "docs/Kisiye_Ozel_TSS_Ozel_Sartlari.pdf"
    ]
    special_loaders = [
        UnstructuredWordDocumentLoader(special_files[0]),
        PyPDFLoader(special_files[1]),
        PyPDFLoader(special_files[2]),
    ]
    special_documents = []
    for loader in special_loaders:
        special_documents.extend(loader.load())

    # Teminat Tabloları Dosyaları
    general_files = [
        "docs/Grup_TeminatTablosu1.pdf",
        "docs/Grup_TeminatTablosu2.pdf",
        "docs/Kisiye_Ozel_Saglik_Sigortasi_Teminat_Tablosu1.pdf",
        "docs/Kisiye_Ozel_Saglik_Sigortasi_Teminat_Tablosu2.pdf",
        "docs/Kisiye_Ozel_TSS_Teminat_Tablosu1.pdf",
        "docs/Kisiye_Ozel_TSS_Teminat_Tablosu2.pdf",
    ]
    general_loaders = [PyPDFLoader(path) for path in general_files]
    general_documents = []
    for loader in general_loaders:
        general_documents.extend(loader.load())

    # Excel Dosyası
    excel_path = "docs/Chatbot_Demo_Sorular.xlsx"
    df = pd.read_excel(excel_path)
    excel_documents = []
    for _, row in df.iterrows():
        question = str(row.get("Soru", "")).strip()
        answer_part1 = str(row.get("Standart Madde 1", "")).strip()
        answer_part2 = str(row.get("Standart Madde 2", "")).strip()
        combined_answer = f"{answer_part1}\n\n{answer_part2}"
        excel_documents.append(Document(page_content=f"Soru: {question}\nCevap: {combined_answer}", metadata={"source": "Excel"}))

    return special_documents, general_documents, excel_documents

# Belgeleri vektör veritabanına dönüştürme
def create_vector_store(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents)
    embedding_model_path = "sentence-transformers/paraphrase-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_path)
    vector_store = FAISS.from_documents(split_docs, embeddings)
    return vector_store

# Önceliklendirilmiş Retriever
class PriorityRetriever(BaseRetriever, BaseModel):
    retrievers: list[BaseRetriever]

    def get_relevant_documents(self, query):
        for retriever in self.retrievers:
            docs = retriever.get_relevant_documents(query)
            if docs:
                return docs
        return []

    async def aget_relevant_documents(self, query):
        for retriever in self.retrievers:
            docs = await retriever.aget_relevant_documents(query)
            if docs:
                return docs
        return []

# Soruya yanıt verme
# Soruya yanıt verme
def get_answer(question, special_vector_store, general_vector_store, excel_vector_store):
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)
    retrievers = [
        special_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10}),
        general_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10}),
        excel_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10}),
    ]
    
    # Her retriever için belgeleri kontrol et ve yazdır
    # for idx, retriever in enumerate(retrievers):
    #     docs = retriever.get_relevant_documents(question)
    #     print(f"Retriever {idx + 1} Retrieved Documents:")
    #     for doc in docs:
    #         print(f"- Content: {doc.page_content[:100]}...")  # İlk 100 karakteri göster
    #         print(f"  Metadata: {doc.metadata}")
    
    # Öncelikli retriever sırasını kullan
    retriever = PriorityRetriever(retrievers=retrievers)
    docs = retriever.get_relevant_documents(question)
    print("Retrieved Documents:", docs)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain.invoke({"query": question})


# Gradio arayüzü
def chatbot_ui():
    special_docs, general_docs, excel_docs = load_and_group_documents()

    # Her grup için vektör veritabanı oluşturma
    special_vector_store = create_vector_store(special_docs)
    general_vector_store = create_vector_store(general_docs)
    excel_vector_store = create_vector_store(excel_docs)

    def chatbot_response(question):
        return get_answer(question, special_vector_store, general_vector_store, excel_vector_store)

    iface = gr.Interface(fn=chatbot_response, inputs="text", outputs="text", title="Sigorta Bilgi Botu")
    iface.launch()

if __name__ == "__main__":
    chatbot_ui()
