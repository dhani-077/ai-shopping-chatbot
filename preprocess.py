import pandas as pd
import numpy as np
import cv2
import pytesseract
import os
from fuzzywuzzy import fuzz
from PIL import Image, ImageEnhance

# Fungsi untuk membersihkan teks
def clean_text(text):
    # Hapus karakter khusus, spasi berlebih, dan ubah ke huruf kecil
    text = text.strip().lower()
    text = ''.join(c for c in text if c.isalnum() or c.isspace())
    return text

# 1. Preprocessing Teks Shopping List
def preprocess_text_shopping_list(input_csv, output_csv):
    df_text = pd.read_csv(input_csv)
    
    # Bersihkan item_name
    df_text['item_name_clean'] = df_text['item_name'].apply(clean_text)
    
    # Standarisasi item umum (contoh mapping sederhana)
    item_mapping = {
        'susu uht': 'susu uht 1l',
        'susu ultra': 'susu uht 1l',
        'beras': 'beras 5kg',
        'sabun cair': 'sabun cair 500ml'
    }
    df_text['item_name_standard'] = df_text['item_name_clean'].map(item_mapping).fillna(df_text['item_name_clean'])
    
    # Simpan hasil
    df_text.to_csv(output_csv, index=False)
    print(f"Hasil preprocessing teks disimpan di: {output_csv}")
    return df_text

# 2. Preprocessing Gambar Shopping List
def preprocess_image(image_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    # Baca dan augmentasi gambar
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Augmentasi: Tingkatkan kontras
    pil_img = Image.fromarray(gray)
    enhancer = ImageEnhance.Contrast(pil_img)
    enhanced_img = enhancer.enhance(2.0)  # Tingkatkan kontras 2x
    enhanced_img = np.array(enhanced_img)
    
    # Denoise
    denoised_img = cv2.fastNlMeansDenoising(enhanced_img, h=30)
    
    # Ekstrak teks dengan pytesseract
    text = pytesseract.image_to_string(denoised_img)
    text_clean = clean_text(text)
    
    # Simpan gambar yang diproses
    output_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_path, denoised_img)
    
    return text_clean, output_path

# 3. Preprocessing Dataset Produk E-commerce (Olist)
def preprocess_ecommerce_data(products_csv, categories_csv, output_csv):
    df_products = pd.read_csv(products_csv)
    df_categories = pd.read_csv(categories_csv)
    
    # Gabungkan dengan kategori terjemahan
    df_products = df_products.merge(df_categories, on='product_category_name', how='left')
    
    # Bersihkan nama kategori
    df_products['product_category_name_english'] = df_products['product_category_name_english'].fillna('unknown')
    df_products['product_category_clean'] = df_products['product_category_name_english'].apply(clean_text)
    
    # Simpan hasil
    df_products.to_csv(output_csv, index=False)
    print(f"Hasil preprocessing produk disimpan di: {output_csv}")
    return df_products

# 4. Mapping Item ke Produk E-commerce
def map_items_to_products(shopping_items, product_list):
    mappings = []
    for item in shopping_items:
        matches = []
        for product in product_list:
            score = fuzz.ratio(item.lower(), product.lower())
            if score > 70:  # Threshold kemiripan
                matches.append((product, score))
        matches = sorted(matches, key=lambda x: x[1], reverse=True)[:3]
        mappings.append({'item': item, 'matches': matches})
    return pd.DataFrame(mappings)

# 5. Generate Link E-commerce
def generate_ecommerce_links(item):
    shopee_link = f"https://shopee.co.id/search?keyword={item.replace(' ', '%20')}"
    tokopedia_link = f"https://www.tokopedia.com/search?st=product&q={item.replace(' ', '%20')}"
    return {'item': item, 'shopee': shopee_link, 'tokopedia': tokopedia_link}

# Main Execution
if __name__ == "__main__":
    # Preprocessing teks shopping list
    df_text = preprocess_text_shopping_list(
        'data/text/shopping_list.csv',
        'data/text/shopping_list_clean.csv'
    )
    
    # Preprocessing gambar shopping list
    text_clean, output_img = preprocess_image(
        'data/images/synthetic_list_1.png',
        'data/images/processed'
    )
    print(f"Teks dari gambar (setelah preprocessing): {text_clean}")
    print(f"Gambar diproses disimpan di: {output_img}")
    
    # Preprocessing dataset Olist
    df_products = preprocess_ecommerce_data(
        'data/ecommerce/olist_products_dataset.csv',
        'data/ecommerce/product_category_name_translation.csv',
        'data/ecommerce/products_clean.csv'
    )
        
    # Generate link e-commerce
    links = [generate_ecommerce_links(item) for item in shopping_items]
    df_links = pd.DataFrame(links)
    df_links.to_csv('data/ecommerce/links.csv', index=False)
    print("\nContoh Link E-commerce:")
    print(df_links.head())