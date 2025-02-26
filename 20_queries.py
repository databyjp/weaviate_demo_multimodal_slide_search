from helpers import text_to_colpali, get_model_and_processor, EMBEDDING_DIR
import numpy as np
import torch

embedding_paths = list(EMBEDDING_DIR.glob("*.npz"))
image_embeddings = []
image_paths = []

_, processor = get_model_and_processor()

for embedding_path in embedding_paths:
    data = np.load(embedding_path, allow_pickle=True)
    embeddings = data["embeddings"]
    filepaths = data["filepaths"]
    for i, filepath in enumerate(filepaths):
        image_embeddings.append(torch.from_numpy(embeddings[i]))
        image_paths.append(filepaths[i])

queries = [
    "Diagrams of Weaviate cluster architecture, with shards, indexes and model integrations.",
    "How much does vector quantization impact memory footprint?",
    "How do I create a new collection in Weaviate?",
    "Vector DBs and spongebob squarepants",
]

query_embeddings = text_to_colpali(queries)

scores = processor.score_multi_vector(query_embeddings, image_embeddings)

print(scores)
print(scores.shape)

