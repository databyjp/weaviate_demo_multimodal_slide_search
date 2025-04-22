from helpers import WEAVIATE_COLLECTION_NAME
import weaviate
from weaviate.classes.query import MetadataQuery
from weaviate.classes.init import Auth
from dotenv import load_dotenv
import os

# client = weaviate.connect_to_local(
#     headers={
#         "X-Cohere-Api-Key": os.environ["COHERE_APIKEY"]
#     }
# )

load_dotenv()

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.environ["APP_WEAVIATE_CLOUD_URL"],
    auth_credentials=Auth.api_key(os.environ["APP_WEAVIATE_CLOUD_APIKEY"]),
    headers={
        "X-Cohere-Api-Key": os.environ["APP_COHERE_API_KEY"]
    }
)

pdfs = client.collections.get(name=WEAVIATE_COLLECTION_NAME)

queries = [
    "Diagrams of Weaviate cluster architecture, with shards, indexes and model integrations.",
    "How much does vector quantization impact memory footprint?",
    "How do I create a new collection in Weaviate?",
    "Vector DBs and spongebob squarepants",
]

for i, query in enumerate(queries):
    print(f"Query: {queries[i]}")
    r = pdfs.query.near_text(
        query=query,
        target_vector="cohere",
        limit=2,
        return_metadata=MetadataQuery(distance=True)
    )

    for o in r.objects:
        print(o.properties)
        print(o.metadata.distance)

client.close()
