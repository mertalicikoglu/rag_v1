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
import pandas as pd
import gradio as gr
import nltk
import pdfplumber

# Gerekli NLTK kaynaklarını indir
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# OpenAI API Anahtarını yükle
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found. Please set it in your .env file.")

# PDF tablolarını ayrıştırma
def parse_table_from_pdf(pdf_path):
    documents = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    row_text = ' | '.join(str(cell) for cell in row if cell)
                    documents.append(Document(page_content=row_text, metadata={"source": pdf_path}))
    return documents

# Dökümanları yükleme ve kategorize etme
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
    general_documents = []
    for path in general_files:
        general_documents.extend(parse_table_from_pdf(path))

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

# Dökümanları anlamlı şekilde bölme
def advanced_chunk_split(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )
    split_docs = []
    for doc in documents:
        split_docs.extend(text_splitter.split_text(doc.page_content))
    return [Document(page_content=chunk, metadata=doc.metadata) for chunk in split_docs]

# Vektör veritabanı oluşturma
def create_vector_store(documents, model_name):
    split_docs = advanced_chunk_split(documents)
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vector_store = FAISS.from_documents(split_docs, embeddings)
    return vector_store

# Öncelikli Retriever
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

# Soruları anlamlandırarak yanıtlama
def enhanced_get_answer(question, special_vector_store, general_vector_store, excel_vector_store):
    # Soruları kategorilere ayır
    category = "teminat" if any(keyword in question.lower() for keyword in ["aşı", "serum", "ilaç"]) else "genel"
    
    retrievers = {
        "special": special_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10}),
        "general": general_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10}),
        "excel": excel_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10}),
    }
    
    if category == "teminat":
        retriever = PriorityRetriever(retrievers=[retrievers["special"], retrievers["general"]])
    else:
        retriever = PriorityRetriever(retrievers=[retrievers["excel"], retrievers["special"], retrievers["general"]])

    llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain.invoke({"query": question})

# Gradio Arayüzü
def chatbot_ui():
    special_docs, general_docs, excel_docs = load_and_group_documents()

    embedding_model_path = "sentence-transformers/all-mpnet-base-v2"
    special_vector_store = create_vector_store(special_docs, embedding_model_path)
    general_vector_store = create_vector_store(general_docs, embedding_model_path)
    excel_vector_store = create_vector_store(excel_docs, embedding_model_path)

    def chatbot_response(question):
        return enhanced_get_answer(question, special_vector_store, general_vector_store, excel_vector_store)

    iface = gr.Interface(fn=chatbot_response, inputs="text", outputs="text", title="Sigorta Bilgi Botu")
    iface.launch()

if __name__ == "__main__":
    chatbot_ui()
