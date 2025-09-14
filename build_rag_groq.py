import os
import getpass
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

# Knowledge base: Olist + database beras
documents = [
    Document(page_content="Dataset Olist e-commerce: Kategori produk seperti grain untuk beras, dairy untuk susu, hygiene untuk sabun. Harga rata-rata beras 5kg: Rp 70.000."),
    Document(page_content="Rekomendasi beras putih: Topi Koki Setra Ramos (Rp 65.000-80.000/5kg, pulen terjangkau), Beras Ngawiti Mas (Rp 64.000-136.000/5kg, bebas kimia)."),
    Document(page_content="Rekomendasi beras merah: Ayana Beras Merah Organik (Rp 80.000-100.000/5kg, kaya serat untuk diet)."),
    Document(page_content="Rekomendasi beras organik: Beras Karya Alam Hijau (Rp 90.000-110.000/5kg, tanpa pewarna). Promo sering di Shopee/Tokopedia."),
    # Tambah lebih banyak dari Olist jika perlu
]

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Split dan buat vector store
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
vectorstore = FAISS.from_documents(texts, embeddings)

# Simpan
vectorstore.save_local("groq_rag_index")
print("Knowledge base RAG Groq siap!")