from helpers import text_to_colpali, get_model_and_processor, EMBEDDING_DIR
from colpali_engine.interpretability import (
    get_similarity_maps_from_embeddings,
    plot_all_similarity_maps,
)
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch

embedding_paths = list(EMBEDDING_DIR.glob("*.npz"))
image_embeddings = []
image_paths = []
device = "mps"

model, processor = get_model_and_processor()

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

# Find the index of the highest similarity score for each query
best_matches = torch.argmax(scores, dim=1)

# Print the best match indices
print("Best match indices:", best_matches)

# Print the best match scores
best_scores = torch.max(scores, dim=1).values
print("Best match scores:", best_scores)

# Print the corresponding image paths for each best match
for query_idx, match_idx in enumerate(best_matches):
    print(f"Query: '{queries[query_idx]}'")
    print(f"Best match: {image_paths[match_idx]}")
    print(f"Score: {scores[query_idx, match_idx]:.4f}")
    print("---")

# Display the best matching images, with heatmap for explainability

# Find top matches for each query
top_k = 3
top_values, top_indices = torch.topk(scores, k=top_k, dim=1)


# Function to visualize top matches with heatmaps
def visualize_top_matches_with_heatmaps(
    query_idx, image_paths, top_indices, top_values, queries, preloaded_image_embeddings
):
    query = queries[query_idx]
    match_indices = top_indices[
        query_idx
    ]  # These are indices into the full list of embeddings
    match_scores = top_values[query_idx]

    print(f"Query: '{query}'")

    # Create a figure to display matches and heatmaps
    fig, axes = plt.subplots(top_k, 2, figsize=(12, 4 * top_k))

    # Process the query
    batch_queries = processor.process_queries([query]).to(device)
    query_embeddings = model.forward(**batch_queries)

    for rank, (score, idx) in enumerate(zip(match_scores, match_indices)):
        # idx is the index in the full image_embeddings list
        img_path = image_paths[idx]
        print(f"  Rank {rank+1}: {img_path} (Score: {score:.4f})")

        # Load image (just for display purposes)
        try:
            image = Image.open(img_path)
        except (FileNotFoundError, IOError):
            print(f"Error: Could not open image file {img_path}")
            image = Image.new("RGB", (224, 224), color=(200, 200, 200))

        # Display the image in first column
        axes[rank, 0].imshow(image)
        axes[rank, 0].set_title(f"Rank {rank+1} - Score: {score:.4f}")
        axes[rank, 0].axis("off")

        # Use preloaded embeddings instead of recomputing them
        # Here's the key change: access the correct embedding directly
        single_image_embedding = preloaded_image_embeddings[idx].unsqueeze(0).to(device)

        # Preprocess image just to get the mask and patches (not for embedding)
        batch_images = processor.process_images([image]).to(device)

        # Get the number of image patches
        n_patches = processor.get_n_patches(
            image_size=image.size, patch_size=model.patch_size
        )

        # Get the tensor mask to filter out embeddings not related to the image
        image_mask = processor.get_image_mask(batch_images)

        # Generate similarity maps using preloaded embeddings
        batched_similarity_maps = get_similarity_maps_from_embeddings(
            image_embeddings=single_image_embedding,
            query_embeddings=query_embeddings,
            n_patches=n_patches,
            image_mask=image_mask,
        )

        # Get avg similarity map across all tokens (for summary view)
        similarity_maps = batched_similarity_maps[0]
        avg_similarity_map = similarity_maps.mean(dim=0)

        # Plot heatmap over image
        axes[rank, 1].imshow(image)
        avg_similarity_map = avg_similarity_map.cpu().to(torch.float32).numpy()
        axes[rank, 1].imshow(avg_similarity_map, cmap="jet", alpha=0.5)
        axes[rank, 1].set_title("Attention Heatmap")
        axes[rank, 1].axis("off")

    plt.tight_layout()
    return fig


# Visualize results for each query
for query_idx in range(len(queries)):
    fig = visualize_top_matches_with_heatmaps(
        query_idx=query_idx,
        image_paths=image_paths,
        top_indices=top_indices,
        top_values=top_values,
        queries=queries,
        preloaded_image_embeddings=image_embeddings,
    )
    fig.savefig(f"query_{query_idx}_results.png", bbox_inches="tight")
    plt.close(fig)

    # Additionally save detailed token-by-token heatmaps for the best match
    best_match_idx = top_indices[query_idx][0]
    best_match_path = image_paths[best_match_idx]

    try:
        image = Image.open(best_match_path)
        batch_images = processor.process_images([image]).to(device)
        batch_queries = processor.process_queries([queries[query_idx]]).to(device)

        # Use preloaded embeddings instead of recomputing
        single_image_embedding = (
            image_embeddings[best_match_idx].unsqueeze(0).to(device)
        )
        query_embeddings = model.forward(**batch_queries)

        n_patches = processor.get_n_patches(
            image_size=image.size, patch_size=model.patch_size
        )
        image_mask = processor.get_image_mask(batch_images)

        batched_similarity_maps = get_similarity_maps_from_embeddings(
            image_embeddings=single_image_embedding,  # Use preloaded embedding
            query_embeddings=query_embeddings,
            n_patches=n_patches,
            image_mask=image_mask,
        )

        similarity_maps = batched_similarity_maps[0]
        query_tokens = processor.tokenizer.tokenize(queries[query_idx])

        plots = plot_all_similarity_maps(
            image=image,
            query_tokens=query_tokens,
            similarity_maps=similarity_maps,
        )

        for token_idx, (fig, ax) in enumerate(plots):
            fig.savefig(f"query_{query_idx}_best_match_token_{token_idx}.png")
            plt.close(fig)
    except Exception as e:
        print(f"Error processing best match for query {query_idx}: {e}")

    print("---")
