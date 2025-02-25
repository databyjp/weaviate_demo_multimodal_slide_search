import numpy as np
from helpers import EMBEDDING_DIR, WEAVIATE_COLLECTION_NAME
from pathlib import Path
import weaviate
from weaviate.classes.config import Property, DataType, Configure
import base64

client = weaviate.connect_to_local()

client.collections.delete(name=WEAVIATE_COLLECTION_NAME)

pdfs = client.collections.create(
    name=WEAVIATE_COLLECTION_NAME,
    properties=[
        Property(name="filepath", data_type=DataType.TEXT),
        Property(name="image", data_type=DataType.BLOB),
    ],
    vectorizer_config=[
        Configure.NamedVectors.none(
            name="pdf_colpali_13",
            vector_index_config=Configure.VectorIndex.hnsw(
                multi_vector=Configure.VectorIndex.MultiVector.multi_vector(),
            ),
        )
    ],
    generative_config=Configure.Generative.cohere(model="command-r"),
)


embedding_paths = list(EMBEDDING_DIR.glob("*.npz"))
embeddings = {}


with pdfs.batch.fixed_size(10) as batch:
    for embedding_path in embedding_paths:
        data = np.load(embedding_path, allow_pickle=True)
        embeddings = data["embeddings"]
        filepaths = data["filepaths"]
        for i, filepath in enumerate(filepaths):
            img_file = Path(filepath)
            batch.add_object(
                properties={
                    "filepath": str(img_file),
                    "image": base64.b64encode(img_file.read_bytes()).decode("utf-8"),
                },
                vector={"pdf_colpali_13": embeddings[i].tolist()},
            )

if pdfs.batch.failed_objects:
    print(pdfs.batch.failed_objects[0].message)

print(len(pdfs))

client.close()
