from helpers import WEAVIATE_COLLECTION_NAME, text_to_colpali
import weaviate
from weaviate.classes.query import MetadataQuery

client = weaviate.connect_to_local()

pdfs = client.collections.get(name=WEAVIATE_COLLECTION_NAME)

queries = [
    "Diagrams of Weaviate cluster architecture, with shards, indexes and model integrations.",
    "How much does vector quantization impact memory footprint?",
    "How do I create a new collection in Weaviate?",
    "Vector DBs and spongebob squarepants",
]


query_embeddings = text_to_colpali(queries).tolist()

for i, query_embedding in enumerate(query_embeddings):
    print(f"Query: {queries[i]}")
    r = pdfs.query.near_vector(
        near_vector=query_embedding,
        target_vector="pdf_colpali",
        limit=2,
        return_metadata=MetadataQuery(distance=True)
    )

    for o in r.objects:
        print(o.properties)
        print(o.metadata.distance)

client.close()
