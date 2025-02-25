from helpers import WEAVIATE_COLLECTION_NAME, text_to_colpali
import weaviate
import base64

client = weaviate.connect_to_local()

pdfs = client.collections.get(name=WEAVIATE_COLLECTION_NAME)

queries = [
    "Diagrams of Weaviate cluster architecture, with shards, indexes and model integrations.",
    "How much does vector quantization impact memory footprint?",
    "HNSW explained in detail, with index parameters."
]


query_embeddings = text_to_colpali(queries)

for i, query_embedding in enumerate(query_embeddings):
    print(f"Query: {queries[i]}")
    r = pdfs.query.near_vector(
        near_vector=query_embedding,
        target_vector="pdf_colpali_13",
        limit=2,
    )

    for o in r.objects:
        print(o.properties)

client.close()
