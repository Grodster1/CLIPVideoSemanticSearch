from src.processing.video_loader import VideoLoader
from src.model.clip_model import CLIPEngine
import numpy as np
import json 
from tqdm import tqdm
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
VIDEO_FILE = 'BigBuckBunny_320x180.mp4'
VIDEO_PATH = os.path.join(PROJECT_ROOT, 'data', 'videos', VIDEO_FILE)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'index')
BATCH_SIZE = 32
SAMPLE_PER_N_SECONDS = 2

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading model")
    model = CLIPEngine()
    print(f"Opening video file {VIDEO_FILE}")
    loader = VideoLoader(VIDEO_PATH, sample_per_n_seconds=SAMPLE_PER_N_SECONDS)
    
    embeddings = []
    metadata = []
    
    batch_images=[]
    batch_timestamps=[]
    
    for (timestamp, img) in tqdm(loader, desc="Proccessing"):
        
        batch_images.append(img)
        batch_timestamps.append(timestamp)
        
        if len(batch_images) >= BATCH_SIZE:
            batch_embeddings = model.get_image_embeddings(batch_images)
            embeddings.append(batch_embeddings)
            metadata.extend(batch_timestamps)
            
            batch_images=[]
            batch_timestamps=[]
            
    if batch_images:
        batch_embeddings = model.get_image_embeddings(batch_images)
        embeddings.append(batch_embeddings)
        metadata.extend(batch_timestamps)
        
    if embeddings:
        final_embeddings = np.vstack(embeddings)
        vectors_path = os.path.join(OUTPUT_DIR, 'vectors.npy')
        metadata_path = os.path.join(OUTPUT_DIR, 'metadata.json')
        
        np.save(vectors_path, final_embeddings)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
            
        print(f"Success, indexes saved in {OUTPUT_DIR}")
    
    else:
        print(f"Generating vectors failed, please check video file: {VIDEO_PATH}")
        
        
if __name__ == "__main__":
    main()