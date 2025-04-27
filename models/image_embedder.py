import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel
import pickle

# ----------------------------------
# CONFIG
# ----------------------------------
KEYFRAMES_DIR = "data/keyframes"  # Directory where frames are stored
IMAGE_EMBEDDINGS_DIR = "embeddings/images"  # Directory to save image embeddings
MODEL_NAME = "openai/clip-vit-base-patch32"  # CLIP model to use
FILENAME_PREFIX = "image_embeddings"  # Prefix for saved embeddings

# ----------------------------------
# CLIP Embedding Class
# ----------------------------------
class ImageEmbedder:
    def __init__(self, model_name=MODEL_NAME, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[INFO] Loading model '{model_name}' on {self.device}...")
        
        # Load the CLIP model and processor
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def embed_images(self, image_paths):
        images = [Image.open(image_path) for image_path in image_paths]
        
        # Preprocess images using CLIP processor
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)

        # Generate image embeddings
        with torch.no_grad():
            embeddings = self.model.get_image_features(**inputs)

        return embeddings.cpu().numpy()

    def save_embeddings(self, embeddings, image_paths, output_dir=IMAGE_EMBEDDINGS_DIR, filename_prefix=FILENAME_PREFIX):
        os.makedirs(output_dir, exist_ok=True)

        # Save .pkl
        pkl_path = Path(output_dir) / f"{filename_prefix}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump({'image_paths': image_paths, 'embeddings': embeddings}, f)
        print(f"[INFO] Saved embeddings to {pkl_path}.")

        # Save .npz
        npz_path = Path(output_dir) / f"{filename_prefix}.npz"
        np.savez(npz_path, image_paths=image_paths, embeddings=embeddings)
        print(f"[INFO] Saved embeddings to {npz_path}.")

# ----------------------------------
# MAIN FUNCTION
# ----------------------------------
def main():
    # Get all image paths from the keyframes directory
    image_paths = sorted(Path(KEYFRAMES_DIR).glob("frame_*.jpg"))
    print(f"[INFO] Found {len(image_paths)} keyframes to process.")

    # Initialize the image embedder
    embedder = ImageEmbedder()

    # Generate embeddings for images
    embeddings = embedder.embed_images([str(image_path) for image_path in image_paths])

    # Save the embeddings
    embedder.save_embeddings(embeddings, [str(image_path) for image_path in image_paths])

if __name__ == "__main__":
    main()
