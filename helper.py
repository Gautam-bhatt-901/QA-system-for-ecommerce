import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline


def scrape_product_data(url):
    '''
    Function to scrape and preprocess the data of the product
    '''

    # Headers to mimic a real browser
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    # Fetch HTML content
    response = requests.get(url, headers=headers)

    soup = BeautifulSoup(response.content, "html.parser")

    # Extract Data
    # 1. Product Name
    product = soup.find("span", class_="VU-ZEz").text.strip()

    # 2. Product price
    header_section = soup.find("div", class_ = "C7fEHH")
    price = header_section.find("div", class_ = "Nx9bqj CxhGGd")
    if price is not None:
        price = price.text.strip()
    else:
        price = 'None'
    # 3. Product Specifications
    specs = {}
    spec_section = soup.find("div", class_="_3Fm-hO")
    if spec_section:
        tables = spec_section.find_all("tr", class_="WJdYP6 row")
        for row in tables:
            key = row.find("td", class_="+fFi1w col col-3-12").text.strip()
            value = row.find("td", class_="Izz52n col col-9-12").text.strip()
            specs[key] = value

    documents = []
    text = f"Poduct: {product}. Price: {price}. "
    text += "Specifications: " + ". ".join([f"{key}: {value}" for key, value in specs.items()])
    documents.append(text)

    return documents

encoder_model = SentenceTransformer('all-MiniLM-L6-v2')
qa_pipeline = pipeline(
    "question-answering",
    model="bert-large-uncased-whole-word-masking-finetuned-squad"
)
def generate_embedding(documents, encoder_model, question):
    # --- Generate Embeddings ---
    embeddings = encoder_model.encode(documents)

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))

    # generating embeddings for the user questions
    query_embedding = encoder_model.encode([question])

    return query_embedding, index, documents
