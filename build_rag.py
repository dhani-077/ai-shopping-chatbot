from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import pickle

# Load dataset
dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")

# Preprocess untuk knowledge base (adaptasi untuk shopping list)
knowledge_base = []
for example in dataset['train']:
    # Adaptasi: Fokus pada intent terkait produk/order
    if example['category'] in ['PRODUCT', 'ORDER', 'SHIPPING']:
        response = example['response'].replace("{{Order Number}}", "your order")  # Clean placeholders
        knowledge_base.append({
            'question': example['instruction'],  # Contoh query seperti "buy milk and rice"
            'answer': response,
            'intent': example['intent'],
            'category': example['category']
        })

# Load model embeddings
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
questions = [item['question'] for item in knowledge_base]
embeddings = model.encode(questions, show_progress_bar=True)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
faiss.normalize_L2(embeddings)
index.add(embeddings.astype('float32'))

# Save
with open('knowledge_base.pkl', 'wb') as f:
    pickle.dump(knowledge_base, f)
faiss.write_index(index, 'ecommerce_index.faiss')
with open('model_name.txt', 'w') as f:
    f.write('sentence-transformers/all-MiniLM-L6-v2')

print("RAG system built!")