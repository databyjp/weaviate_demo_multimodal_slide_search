import streamlit as st
import numpy as np
import torch
from pathlib import Path
import os
from PIL import Image
from helpers import text_to_colpali, get_model_and_processor, EMBEDDING_DIR

# Set page title and layout
st.set_page_config(page_title="Multimodal image search", layout="wide")

st.title("Image Search")
st.markdown(
    "Search for images using natural language queries with ColPali (multi-modal, multi-dimensional vectorizer model)"
)

# Load embeddings only once on app startup
# Load the model and processor only once
@st.cache_resource
def load_model_and_processor():
    model, processor = get_model_and_processor()
    return model, processor


# Load embeddings only once on app startup
@st.cache_resource
def load_embeddings():
    embedding_paths = list(EMBEDDING_DIR.glob("*.npz"))
    image_embeddings = []
    image_paths = []

    for embedding_path in embedding_paths:
        data = np.load(embedding_path, allow_pickle=True)
        embeddings = data["embeddings"]
        filepaths = data["filepaths"]
        for i, filepath in enumerate(filepaths):
            image_embeddings.append(torch.from_numpy(embeddings[i]))
            image_paths.append(filepath)

    return image_embeddings, image_paths


# Initialize the model, processor, and embeddings at app startup
model, processor = load_model_and_processor()
image_embeddings, image_paths = load_embeddings()


# Function to perform search
def search_images(query, top_k=6):
    # Use the globally loaded model, processor and embeddings
    # No need to reload them for each search

    # Convert query to embedding
    query_embedding = text_to_colpali(texts=[query], model=model, processor=processor)

    # Calculate similarity scores
    scores = processor.score_multi_vector(query_embedding, image_embeddings)

    # Find the index of the highest similarity score for each query
    best_matches = torch.argsort(scores, dim=1, descending=True).squeeze(0)

    # Get top k results
    top_k_indices = best_matches[:top_k]

    results = []
    for idx in top_k_indices:
        results.append({"path": image_paths[idx], "score": scores[0][idx].item()})
    return results


# Create the sidebar for input
with st.sidebar:
    st.header("Search Settings")

    # Text input
    query = st.text_input(
        "Enter your query", value="HNSW index parameters"
    )

    # Number of results slider
    num_results = st.slider("Number of results", min_value=1, max_value=12, value=6)

    # Search button
    search_button = st.button("Search")

    # Example queries
    st.subheader("Example queries")
    example_queries = [
        "Weaviate cluster architecture",
        "Memory savings and costs",
        "How to create a new collection",
        "Vector DBs and spongebob squarepants",
    ]

    for example in example_queries:
        if st.button(example[:40] + "..." if len(example) > 40 else example):
            query = example
            search_button = True

# Main area for displaying results
if search_button or query:
    with st.spinner("Searching for images..."):
        results = search_images(query, top_k=num_results)

    st.subheader(f"Results for: '{query}'")

    # Create a grid layout for the results
    cols = 3
    rows = (num_results + cols - 1) // cols

    for row in range(rows):
        columns = st.columns(cols)
        for col in range(cols):
            idx = row * cols + col
            if idx < len(results):
                result = results[idx]
                with columns[col]:
                    try:
                        # Display the image if it exists
                        image_path_str = result["path"]
                        image_path = Path(image_path_str)
                        if image_path.exists():
                            img = Image.open(image_path)
                            st.image(img, caption=f"Score: {result['score']:.4f}")
                            st.caption(f"Path: {image_path_str}")
                        else:
                            st.error(f"Image not found: {image_path}")
                    except Exception as e:
                        st.error(f"Error loading image: {str(e)}")
else:
    st.info("Enter a query in the sidebar and click 'Search' to find images.")

# Add helpful information at the bottom
st.markdown("---")
st.markdown(
    """
### About ColPali
ColPali is a multi-modal, multi-vector model based on PaliGemma and CoLBERT. It allows for semantic search across text and images.
This application uses pre-computed embeddings to find the most relevant images for your query.
"""
)
