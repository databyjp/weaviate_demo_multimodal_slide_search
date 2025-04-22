from helpers import IMG_DIR, WEAVIATE_COLLECTION_NAME
import weaviate
from weaviate.classes.config import Property, DataType, Configure
import base64
import os

client = weaviate.connect_to_local(
    headers={
        "X-Cohere-Api-Key": os.environ["COHERE_APIKEY"]
    }
)

client.collections.delete(name=WEAVIATE_COLLECTION_NAME)

pdfs = client.collections.create(
    name=WEAVIATE_COLLECTION_NAME,
    properties=[
        Property(name="filepath", data_type=DataType.TEXT),
        Property(name="image", data_type=DataType.BLOB),
    ],
    vectorizer_config=[
        Configure.NamedVectors.multi2vec_cohere(
            name="cohere",
            model="embed-v4.0",
            image_fields=["image"]
        )
    ],
    generative_config=Configure.Generative.cohere(model="command-r"),
)


img_paths = list(IMG_DIR.glob("*.png"))


with pdfs.batch.fixed_size(10) as batch:
    for img_path in img_paths:
        batch.add_object(
            properties={
                "filepath": str(img_path.name),
                "image": base64.b64encode(img_path.read_bytes()).decode("utf-8"),
            },
        )

if pdfs.batch.failed_objects:
    print(pdfs.batch.failed_objects[0].message)

print(len(pdfs))

client.close()
