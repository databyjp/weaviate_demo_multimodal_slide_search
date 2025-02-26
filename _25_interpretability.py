from helpers import text_to_colpali, get_model_and_processor, EMBEDDING_DIR, OUT_DIR
from colpali_engine.interpretability import (
    get_similarity_maps_from_embeddings,
    plot_all_similarity_maps,
)
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import scipy.ndimage as ndimage


def generate_explainability_plots(processor, model, image_path, image_embedding, query, query_embedding):
    # Load the image
    image = Image.open(image_path)

    # Use the precomputed query embeddings from text_to_colpali
    query_embeddings = query_embedding.unsqueeze(0).to(device)

    # Preprocess inputs
    batch_queries = processor.process_queries([query]).to(device)

    # Use preloaded embeddings
    single_image_embedding = image_embedding.unsqueeze(0).to(device)
    batch_images = processor.process_images([image]).to(device)

    # Get the number of image patches
    n_patches = processor.get_n_patches(
        image_size=image.size, patch_size=model.patch_size
    )
    image_mask = processor.get_image_mask(batch_images)

    # Generate the similarity maps
    batched_similarity_maps = get_similarity_maps_from_embeddings(
        image_embeddings=single_image_embedding,
        query_embeddings=query_embeddings,
        n_patches=n_patches,
        image_mask=image_mask,
    )

    # Get the similarity map for our (only) input image
    similarity_maps = batched_similarity_maps[0]

    # Tokenize the query
    query_tokens = processor.tokenizer.tokenize(query)

    # Plot and save the similarity maps for each query token
    plots = plot_all_similarity_maps(
        image=image,
        query_tokens=query_tokens,
        similarity_maps=similarity_maps,
    )

    explainability_dir = OUT_DIR / "explainability"
    explainability_dir.mkdir(parents=True, exist_ok=True)

    for idx, (fig, ax) in enumerate(plots):
        fig.savefig(explainability_dir / f"{query}_similarity_map_{idx}.png")
    plt.close(fig)

    # Create and save average heatmap
    # Compute the average across all token maps
    average_map = torch.mean(similarity_maps, dim=0).cpu().numpy()

    # Get image dimensions for proper overlay
    img_array = np.array(image)
    img_height, img_width = img_array.shape[:2]

    # Resize the average map to match image dimensions
    resized_map = ndimage.zoom(
        average_map,
        (img_height / average_map.shape[0], img_width / average_map.shape[1]),
        order=0,
    )

    # Get the size from one of the individual plots for consistency
    example_fig, _ = plots[0]
    fig_width, fig_height = example_fig.get_size_inches()

    # Create a new figure with matching dimensions
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Display the original image
    ax.imshow(img_array)

    # Overlay the resized average heatmap
    heatmap = ax.imshow(resized_map, cmap="Oranges", alpha=0.5)

    # Add a colorbar
    plt.colorbar(heatmap, ax=ax, label="Average similarity")

    # Set the title
    ax.set_title(f"Average similarity for: {query}")

    # Remove axis ticks for cleaner visualization
    ax.set_xticks([])
    ax.set_yticks([])

    # Save the average heatmap
    fig.savefig(explainability_dir / f"{query}_average_heatmap.png")
    plt.close(fig)


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

# For each query, pass the corresponding embedding from text_to_colpali
for i, query in enumerate(queries):
    generate_explainability_plots(
        processor=processor,
        model=model,
        image_path=image_paths[best_matches[i]],
        image_embedding=image_embeddings[best_matches[i]],
        query=query,
        query_embedding=query_embeddings[i],
    )
