import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
import warnings

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Function to clean HTML tags
def clean_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

# Load data
print("Loading dataset...")
dataset = "fashion.csv"  # Updated dataset file name
myntra_fashion_products_df = pd.read_csv(dataset)
myntra_fashion_products_df = myntra_fashion_products_df.drop(['img', 'p_attributes'], axis=1)
print(f"Dataset loaded with {myntra_fashion_products_df.shape[0]} rows and {myntra_fashion_products_df.shape[1]} columns.")

# Clean HTML in 'description' field
print("Cleaning HTML tags from descriptions...")
myntra_fashion_products_df['description'] = myntra_fashion_products_df['description'].apply(clean_html)

# Load Sentence Transformer model
print("Loading Sentence Transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

def create_combined_text(row):
    return f"Name: {row['name']}, Description: {row['description']}, Products: {row['products']}, Price: {row['price']}, Colour: {row['colour']}, Brand: {row['brand']}"

print("Generating combined text...")
myntra_fashion_products_df['combined_text'] = myntra_fashion_products_df.apply(create_combined_text, axis=1)

print("Generating embeddings...")
product_embeddings = model.encode(myntra_fashion_products_df['combined_text'].tolist(), convert_to_tensor=True)

# Convert embeddings to numpy array
product_embeddings_np = np.array(product_embeddings)

# Set up FAISS index for similarity search
print("Setting up FAISS index...")
index = faiss.IndexFlatL2(product_embeddings_np.shape[1])
index.add(product_embeddings_np)

# Save embeddings and FAISS index to pickle files
print("Saving embeddings and FAISS index to pickle files...")
with open('product_embeddings.pkl', 'wb') as f:
    pickle.dump(product_embeddings_np, f)

with open('faiss_index.pkl', 'wb') as f:
    pickle.dump(index, f)

print("Embeddings and FAISS index saved successfully.")
