import time
import os
from src.processing.video_loader import VideoLoader
from src.model.clip_model import CLIPEngine

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
VIDEO_FILE = os.path.join(PROJECT_ROOT, 'data', 'videos', 'BigBuckBunny_320x180.mp4')

try:
    
    clip = CLIPEngine()
    loader = VideoLoader(VIDEO_FILE, sample_per_n_seconds=2)
    
    batch_images=[]
    batch_timestamps=[]
    
    print(f"FPS: {loader.fps}")
    print(f"Total frames to analyze: {len(loader)}")
    
    print("Starting analyzing:")
    
    for i, (timestamp, img) in enumerate(loader):
        print(f"Frame {i}: Time {timestamp:.2f}s | Image size: {img.size}")
        
        batch_images.append(img)
        batch_timestamps.append(timestamp)
        
        if i == 2:
            img.show()
            
        if i >=4:
            break
        
    print(f"Processing batches: {len(batch_images)}")
    
    start_time = time.time()
    vectors = clip.get_image_embeddings(batch_images)
    end_time = time.time()
    
    print(f"Time elapsed for operation: {end_time-start_time:.4f} seconds")
    print(f"Output matrix size: {vectors.shape}")
    
    
        
except Exception as e:
    print(f"Error occured: {e}")
    