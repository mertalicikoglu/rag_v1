# Sigorta Bilgi Botu Dokümantasyonu

## 1. Genel Açıklama
Bu proje, sigorta ile ilgili belgeleri işleyerek doğal dil işleme (NLP) teknikleri kullanarak sigorta teminatları, özel şartlar ve sıkça sorulan sorular hakkında bilgi sağlayan bir chatbot uygulamasıdır. Python dili ile geliştirilmiş olup, **LangChain**, **FAISS**, **HuggingFace Embeddings**, **Gradio**, ve **pdfplumber** gibi kütüphanelerden faydalanmaktadır.

## 2. Gerekli Bağımlılıklar
Projenin çalışması için aşağıdaki Python kütüphanelerinin yüklenmesi gerekmektedir:

```bash
pip install langchain gradio pandas pdfplumber nltk pydantic openai python-dotenv faiss-cpu
```

Ayrıca **NLTK** kütüphanesinin bazı veri setleri indirilmelidir:

```python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
```

## 3. OpenAI API Anahtarı Yükleme

Proje, OpenAI API anahtarını **.env** dosyasından okumaktadır. `.env` dosyanızın içeriğinde şu şekilde bir tanımlama olmalıdır:

```
OPENAI_API_KEY=your_openai_api_key_here
```

Eğer anahtar bulunamazsa, sistem hata verecektir.

## 4. PDF Tablolarını Ayrıştırma

**pdfplumber** kütüphanesi kullanılarak PDF dosyalarındaki tablo verileri metin olarak çıkarılmaktadır:

```python
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
```

## 5. Dökümanları Yükleme ve Kategorize Etme
Proje, üç farklı kategoriye ayrılmış sigorta belgelerini işler:

- **Özel Şartlar**: `.docx` ve `.pdf` dosyalarından yüklenir.
- **Teminat Tabloları**: PDF dosyalarındaki tablolar ayrıştırılır.
- **Sıkça Sorulan Sorular (Excel)**: Excel dosyası işlenerek soru-cevap formatına dönüştürülür.

Dökümanlar şu fonksiyon ile yüklenir:

```python
def load_and_group_documents():
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
    
    general_files = [
        "docs/Grup_TeminatTablosu1.pdf",
        "docs/Grup_TeminatTablosu2.pdf",
        "docs/Kisiye_Ozel_Saglik_Sigortasi_Teminat_Tablosu1.pdf",
    ]
    general_documents = []
    for path in general_files:
        general_documents.extend(parse_table_from_pdf(path))
    
    df = pd.read_excel("docs/Chatbot_Demo_Sorular.xlsx")
    excel_documents = []
    for _, row in df.iterrows():
        question = str(row.get("Soru", "")).strip()
        answer = f"{row.get('Standart Madde 1', '')}\n\n{row.get('Standart Madde 2', '')}"
        excel_documents.append(Document(page_content=f"Soru: {question}\nCevap: {answer}", metadata={"source": "Excel"}))
    
    return special_documents, general_documents, excel_documents
```

## 6. Dökümanları Parçalama

**RecursiveCharacterTextSplitter** kullanılarak dökümanlar anlamlı parçalara bölünmektedir:

```python
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
```

## 7. Vektör Veritabanı Oluşturma

FAISS kullanılarak vektör veritabanı oluşturulmaktadır:

```python
def create_vector_store(documents, model_name):
    split_docs = advanced_chunk_split(documents)
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vector_store = FAISS.from_documents(split_docs, embeddings)
    return vector_store
```

## 8. Sorgu İşleme

Girilen sorular, ilgili kategorilere yönlendirilerek en uygun kaynaktan yanıtlanır:

```python
def enhanced_get_answer(question, special_vector_store, general_vector_store, excel_vector_store):
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
```

## 9. Gradio Arayüzü

Chatbot arayüzü **Gradio** kullanılarak oluşturulmuştur:

```python
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
```

## 10. Çalıştırma

Projeyi çalıştırmak için aşağıdaki komut kullanılır:

```bash
python chatbot.py
```

