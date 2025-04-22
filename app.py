import streamlit as st
from pathlib import Path
from PIL import Image
from helpers import (
    render_svg_file,
    WEAVIATE_COLLECTION_NAME,
    IMG_DIR
)
import weaviate
from weaviate.classes.query import MetadataQuery
from weaviate.classes.init import Auth
import os

# client = weaviate.connect_to_local(
#     headers={
#         "X-Cohere-Api-Key": os.environ["COHERE_APIKEY"]
#     }
# )

# Not necessary as secrets loaded from .streamlit/secrets.toml
# from dotenv import load_dotenv
# load_dotenv()

# Initialize session state for query
if 'query' not in st.session_state:
    st.session_state.query = ""

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.environ["APP_WEAVIATE_CLOUD_URL"],
    auth_credentials=Auth.api_key(os.environ["APP_WEAVIATE_CLOUD_APIKEY"]),
    headers={
        "X-Cohere-Api-Key": os.environ["APP_COHERE_API_KEY"]
    }
)

st.title("Weaviate + Multimodal Image Search")
st.markdown(
    "Search for images using natural language queries with a multi-modal vectorizer model"
)


# Function to perform search
def search_images(query, weaviate_client, top_k=6):
    pdfs = weaviate_client.collections.get(name=WEAVIATE_COLLECTION_NAME)

    response = pdfs.query.near_text(
        query=query,
        target_vector="cohere",
        limit=top_k,
        return_metadata=MetadataQuery(distance=True),
    )

    return response


# Create the sidebar for input
with st.sidebar:
    st.header("Search Settings")

    # Function to reset query
    def reset_query():
        st.session_state.query = ""

    # Text input using session state
    query = st.text_input("Enter your query", key="query")

    # Search button
    col1, col2 = st.columns(2)
    with col1:
        search_button = st.button("Search")
    with col2:
        reset_button = st.button("Reset", on_click=reset_query)

    # Example queries
    st.subheader("Example queries")
    example_queries = [
        "Weaviate cluster architecture",
        "Memory savings and costs",
        "How to create a new collection",
        "Vector DBs and spongebob squarepants",
    ]

    # Function to set example query
    def set_example_query(example):
        st.session_state.query = example

    for example in example_queries:
        if st.button(example[:40] + "..." if len(example) > 40 else example,
                     on_click=set_example_query,
                     args=(example,)):
            search_button = True

    st.subheader("Query settings")
    # Number of results slider
    num_results = st.slider("Number of results", min_value=1, max_value=12, value=6)

    # Display the SVG logo with specified dimensions
    st.write(
        "Powered by:", render_svg_file(
            "assets/weaviate-logo-square-dark.svg", width="80px", height="80px"
        ),
        unsafe_allow_html=True,
    )

# Main area for displaying results
if (search_button or len(query) > 0) and query.strip():
    with st.spinner("Searching for images..."):
        results = search_images(query, weaviate_client=client, top_k=num_results)

    st.subheader(f"Results for: '{query}'")

    # Create a grid layout for the results
    cols = 3
    rows = (num_results + cols - 1) // cols

    for row in range(rows):
        columns = st.columns(cols)
        for col in range(cols):
            idx = row * cols + col
            if idx < len(results.objects):
                result = results.objects[idx]
                with columns[col]:
                    try:
                        # Display the image if it exists
                        image_path_str = result.properties["filepath"]
                        image_path = Path(IMG_DIR) / Path(image_path_str)
                        if image_path.exists():
                            img = Image.open(image_path)
                            st.image(
                                img, caption=f"Distance: {result.metadata.distance:.2f}"
                            )
                            st.caption(f"Path: {image_path_str}")
                        else:
                            st.error(f"Image not found: {image_path}")
                    except Exception as e:
                        st.error(f"Error loading image: {str(e)}")
elif search_button and not query.strip():
    st.error("Please enter a search query before clicking the Search button.")
else:
    st.info("Have you ever tried to find a particular slide? It's a huge pain. \n\nBut... not with **multimodal search!** \n\nEnter a query in the sidebar and click 'Search' to find images.")
