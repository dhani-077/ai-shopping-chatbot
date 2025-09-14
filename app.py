import streamlit as st
import pandas as pd
import cv2
import pytesseract
import numpy as np
from PIL import Image, ImageEnhance
from fuzzywuzzy import fuzz
from skimage.filters import threshold_otsu
import re
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

# Set API key Groq
load_dotenv()
if "GROQ_API_KEY" not in os.environ:
    st.error("Set GROQ_API_KEY di environment variable!")
    st.stop()

# Load embeddings dan vector store
@st.cache_resource
def load_rag_groq():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("groq_rag_index", embeddings, allow_dangerous_deserialization=True)
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.7)
    
    # Prompt template untuk RAG
    template = """Gunakan konteks berikut untuk jawab query. Jika query tentang shopping list, ekstrak item dan quantity. Jika rekomendasi, saran produk lokal dengan link promo.
    Konteks: {context}
    Query: {question}
    Jawaban:"""
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

qa_chain = load_rag_groq()

# Fungsi preprocessing teks
def clean_text(text):
    text = text.strip().lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# Standarisasi item
def standardize_items(items):
    item_mapping = {
        'susu uht': 'susu uht 1l',
        'susu ultra': 'susu uht 1l',
        'beras': 'beras 5kg',
        'sabun cair': 'sabun cair 500ml',
        'minyak goreng': 'minyak goreng 1l',
        'telur': 'telur ayam 1kg'
    }
    return [item_mapping.get(item, item) for item in items]

# Preprocessing gambar
def preprocess_image(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pil_img = Image.fromarray(gray)
    enhancer = ImageEnhance.Contrast(pil_img)
    enhanced_img = enhancer.enhance(2.0)
    gray_enhanced = np.array(enhanced_img)
    thresh = threshold_otsu(gray_enhanced)
    binary = gray_enhanced > thresh
    denoised_img = cv2.fastNlMeansDenoising(gray_enhanced, h=30)
    binary_img = (binary * 255).astype(np.uint8)
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(binary_img, config=custom_config)
    return clean_text(text)

# Generate link e-commerce reguler
def generate_ecommerce_links(item):
    shopee_link = f"https://shopee.co.id/search?keyword={item.replace(' ', '%20')}"
    tokopedia_link = f"https://www.tokopedia.com/search?st=product&q={item.replace(' ', '%20')}"
    return f"- **{item}**: [Shopee]({shopee_link}) | [Tokopedia]({tokopedia_link})"

# Generate link promo
def generate_promo_links(item):
    shopee_promo = f"https://shopee.co.id/search?keyword={item.replace(' ', '%20')}&filter=discount"
    tokopedia_promo = f"https://www.tokopedia.com/search?st=product&q={item.replace(' ', '%20')}&tab=promo"
    return f"- **{item} (Promo)**: [Shopee Promo]({shopee_promo}) | [Tokopedia Promo]({tokopedia_promo})"

# Load dataset Olist untuk fallback
@st.cache_data
def load_olist_data():
    df_products = pd.read_csv('data/ecommerce/olist_products_dataset.csv')
    df_categories = pd.read_csv('data/ecommerce/product_category_name_translation.csv')
    df_products = df_products.merge(df_categories, on='product_category_name', how='left')
    df_products['product_category_clean'] = df_products['product_category_name_english'].fillna('unknown').apply(clean_text)
    return df_products['product_category_clean'].tolist()

# Mapping item ke produk (fallback)

#def map_items_to_products(shopping_items, product_list):
#    mappings = []
#    for item in shopping_items:
#        matches = []
#        for product in product_list:
#            score = fuzz.token_sort_ratio(item.lower(), product.lower())
#            if score > 65:
#                matches.append((product, score))
#        matches = sorted(matches, key=lambda x: x[1], reverse=True)[:3]
#        mappings.append({'item': item, 'matches': matches})
#    return mappings


# Antarmuka Streamlit
st.title("AI Chatbot Shopping List")

# Inisialisasi session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "item_history" not in st.session_state:
    st.session_state.item_history = []

# Tampilkan chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Opsi input
input_type = st.radio("Pilih jenis input shopping list:", ("Teks", "Gambar"))

shopping_list_text = ""
items = []
if input_type == "Teks":
    shopping_list_text = st.text_area("Masukkan shopping list (satu item per baris):")
    if shopping_list_text:
        # Gunakan RAG untuk parse
        rag_response = qa_chain.run(shopping_list_text)
        items = re.findall(r'(\w+\s*\w*)\s*(\d+)?', rag_response)  # Ekstrak item dari respons RAG
        items = [f"{item[0]} {item[1]}" if item[1] else item[0] for item in items]
else:
    uploaded_image = st.file_uploader("Upload gambar shopping list:", type=["jpg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Gambar yang diupload")
        shopping_list_text = preprocess_image(image)
        st.write("Teks yang diekstrak dari gambar:")
        st.text(shopping_list_text)
        # Gunakan RAG untuk parse
        rag_response = qa_chain.run(shopping_list_text)
        items = re.findall(r'(\w+\s*\w*)\s*(\d+)?', rag_response)
        items = [f"{item[0]} {item[1]}" if item[1] else item[0] for item in items]

# Proses shopping list
if st.button("Proses Shopping List") and shopping_list_text:
    items_standard = standardize_items(items)
    st.session_state.item_history.extend(items_standard)
    
    # Load Olist untuk mapping fallback
    # product_list = load_olist_data()
    # mappings = map_items_to_products(items_standard, product_list)
    
    # Generate link
    links = [generate_ecommerce_links(item) for item in items_standard]
    if "promo" in shopping_list_text.lower() or "diskon" in shopping_list_text.lower():
        links.extend([generate_promo_links(item) for item in items_standard])
    
    # Respons dengan RAG Groq
    rag_query = f"Proses shopping list: {shopping_list_text}. Berdasarkan history: {', '.join(set(st.session_state.item_history))}. Berikan link dan mapping."
    bot_response = qa_chain.run(rag_query)
    
    #bot_response += "\n\n**Mapping ke produk (fallback):**"
    #for mapping in mappings:
    #    matches = mapping['matches']
    #    if matches:
    #        bot_response += f"\n- **{mapping['item']}**: {', '.join([f'{m[0]} ({m[1]}%)' for m in matches])}"
    #    else:
    #        bot_response += f"\n- **{mapping['item']}**: Tidak ditemukan match."
    
    bot_response += "\n\nIngin cari promo lebih lanjut atau tanya rekomendasi?"
    
    st.session_state.messages.append({"role": "user", "content": "Shopping list diproses."})
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    
    with st.chat_message("assistant"):
        st.markdown(bot_response)

# Input chat tambahan dengan RAG Groq
if prompt := st.chat_input("Tanya sesuatu tentang belanja (misalnya: rekomendasi beras):"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Respons dengan RAG Groq
    rag_query = f"Query: {prompt}. Berdasarkan history: {', '.join(set(st.session_state.item_history))}. Sertakan link promo jika relevan."
    bot_reply = qa_chain.run(rag_query)
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.markdown(bot_reply)