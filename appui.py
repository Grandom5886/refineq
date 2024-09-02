import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
from openai import OpenAI
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load data
print("Loading dataset...")
dataset = "fashion.csv"  # Updated dataset file name
myntra_fashion_products_df = pd.read_csv(dataset)
myntra_fashion_products_df = myntra_fashion_products_df.drop(['p_attributes'], axis=1)
print(f"Dataset loaded with {myntra_fashion_products_df.shape[0]} rows and {myntra_fashion_products_df.shape[1]} columns.")

# Clean HTML in 'description' field
print("Cleaning HTML tags from descriptions...")
def clean_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

myntra_fashion_products_df['description'] = myntra_fashion_products_df['description'].apply(clean_html)

# Load embeddings and FAISS index
print("Loading embeddings and FAISS index...")
with open('product_embeddings.pkl', 'rb') as f:
    product_embeddings_np = pickle.load(f)

with open('faiss_index.pkl', 'rb') as f:
    index = pickle.load(f)

# Load Sentence Transformer model
print("Loading Sentence Transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_search(query, k=3):
    # Generate embedding for the query
    print("Generating query embedding...")
    query_embedding = model.encode([query], convert_to_tensor=True)
    query_embedding_np = np.array(query_embedding)

    # Perform search
    print("Performing search...")
    distances, indices = index.search(query_embedding_np, k)
    return distances, indices


client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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"
)

prompt = """
You are an apparel recommender agent for an Indian apparel company. Your job is to suggest different types of apparel based on the user's query. You can understand the occasion and recommend the correct apparel items for the occasion if applicable, or just output the specific apparel items if the user is already very specific. Below are a few examples with reasons as to why the particular item is recommended:
1. User question: "Show me blue shirts"
   Response: "blue shirts"
   Reason for recommendation: User is already specific in their query, nothing to recommend.
2. User question: "What can I wear for an office party?"
   Response: "semi formal dress, suit, dress shirt, heels or dress shoes"
   Reason for recommendation: Recommend apparel choices based on occasion.
3. User question: "I am doing shopping for trekking in mountains. What do you suggest?"
   Response: "heavy jacket, jeans, boots, windbreaker, sweater"
   Reason for recommendation: Recommend apparel choices based on occasion.
4. User question: "What should one person wear for their child's graduation ceremony?"
   Response: "Dress or pantsuit, dress shirt, heels or dress shoes"
   Reason for recommendation: Recommend apparel choices based on occasion.
5. User question: "Sunflower dress"
   Response: "sunflower dress, yellow"
   Reason for recommendation: User is specific about their query, nothing to recommend.
6. User question: "What's the price of the 2nd item?"
   Response: "##detail##"
   Reason for recommendation: User is asking for information related to a product already recommended, in which case you should only return '##detail##'.
7. User question: "What is the price of the 4th item in the list?"
   Response: "##detail##"
   Reason for recommendation: User is asking for information related to a product already recommended, in which case you should only return '##detail##'.
8. User question: "What are their brand names?"
   Response: "##detail##"
   Reason for recommendation: User is asking for information related to a product already recommended, in which case you should only return '##detail##'.
9. User question: "Show me more products with a similar brand to this item."
   Response: "brand name of the item"
   Reason for recommendation: User is asking for similar products; return the original product.
10. User question: "Do you have more red dresses in similar patterns?"
    Response: "name of that red dress"
    Reason for recommendation: User is asking for similar products; return the original product.
11. User question: "Show me some tops from H&M"
   Response: "H&M brand, H&M tops,"
   Reason for recommendation: User is asking for clothes from specific brand and category.
Only suggest the apparels or only relevant information. Do not return anything else, which is not related to fashion search.
"""

def get_openai_context(prompt:str, chat_history:str) -> str:
    """Get context from OpenAI model."""
    response = client.chat.completions.create(
      model="meta/llama-3.1-405b-instruct",
      messages=[
          {"role":"system","content":prompt},
          {"role": "user", "content": chat_history}
      ],
      temperature=1,
      max_tokens=500
    )
    return response.choices[0].message.content

def generate_query_embeddings(user_message:str):
    """Generate user message embeddings."""
    openai_context = get_openai_context(prompt, user_message)
    query_emb = model.encode(user_message + " " + openai_context).astype('float32').reshape(1, -1)
    return query_emb

def query_product_names_from_embeddings(query_emb, top_k):
    query_embedding_np = np.array(query_emb)
    distances, indices = index.search(query_embedding_np, k=top_k)
    top_products = myntra_fashion_products_df.iloc[indices[0]]
    return top_products

def get_recommendations(user_message:str, top_k=5):
    """Get recommendations."""
    embeddings = generate_query_embeddings(user_message)
    p_names = query_product_names_from_embeddings(embeddings, top_k)
    return p_names

second_llm_prompt = (
    """
    You are a chatbot assistant helps user with product search from fashion products ecommerce.
    You are provided with users query,llm refined query and some apparel recommendations from the brand's database.
    Your job is to present the most relevant items from the data given to you.
    wish the user, if there is some special occation.
    then give product name alone - and reason.
    If user is asking a clarifying question about one of the recommended item, like what is it's price or brand, then answer that question from its description.
    Do not answer anything else apart from apparel recommendation or search from the company's database.
    """
)

# Streamlit app
st.markdown("""
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20;
            padding: 0;
            background-color: #f4f4f4;
        }
        .container {
            width: 80%;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .header h1 {
            font-size: 36px;
            color: #333;
        }
        .search-bar {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-bottom: 40px;
        }
        .search-bar input[type="text"] {
            width: 60%;
            padding: 15px;
            border: 1px solid blue;
            border-radius: 25px;
            font-size: 16px;
            margin-right: 10px;
        }
        .search-bar button {
            padding: 15px 30px;
            border: none;
            background-color: #007bff;
            color: white;
            border-radius: 25px;
            font-size: 16px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        .search-bar button:hover {
            background-color: #0056b3;
        }
        .product-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 20px;
        }
        .product-item {
            background-color: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
        }
        .product-item:hover {
            transform: scale(1.05);
        }
        .product-item img {
            width: 100%;
            height: auto;
            border-bottom: 1px solid #ddd;
        }
        .product-item .product-info {
            padding: 10px;
            text-align: center;
        }
        .product-item .product-name {
            font-size: 13px;
            margin-bottom: 5px;
            overflow: hidden;
            text-overflow: ellipsis;
            display: -webkit-box;
            -webkit-line-clamp: 2; /* Number of lines to show */
            -webkit-box-orient: vertical;
            white-space: normal;
            min-height: 2.6em; /* Ensures it always occupies space for two lines */
        }
        .product-item .product-price {
            font-size: 16px;
        }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<div class='container'>", unsafe_allow_html=True)

st.markdown("""
    <div class='header'>
        <h1>Mumbai Marines: LLM-based Product Search and Image Caption Generation</h1>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div class='search-bar'>", unsafe_allow_html=True)

query = st.text_input(label="", placeholder="Type your query here...", label_visibility="collapsed")

st.button("Search")
st.markdown("</div>", unsafe_allow_html=True)


if query:
    # refined_query = get_openai_context(prompt, query)
    # response = get_recommendations(refined_query)
    # message = get_openai_context(second_llm_prompt, f"User question = query : '{query}' and llm refined query : '{refined_query}', our recommendations = {response}")

    # st.markdown(f"Refined Query: {refined_query}")
    # st.markdown(f"{message}\n\n")

    distances, indices = semantic_search(query, k=5)
    top_products = myntra_fashion_products_df.iloc[indices[0]]

    st.markdown("<div class='product-grid'>", unsafe_allow_html=True)

    cols = st.columns(3)

    for idx, row in top_products.iterrows():
        col = cols[idx % 3]
        with col:
            st.markdown(f"""
                <div class='product-item'>
                    <img src='{row['img']}' alt='{row['name']}'>
                    <div class='product-info'>
                        <div class='product-name'>{row['name']}</div>
                        <div class='product-price'>Rs. {row['price']}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
