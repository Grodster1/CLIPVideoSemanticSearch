import sys
import os
from src.processing.video_loader import VideoLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
VIDEO_FILE = os.path.join(PROJECT_ROOT, 'data', 'videos', 'BigBuckBunny_320x180.mp4')

try:
    loader = VideoLoader(VIDEO_FILE, sample_per_n_seconds=2)
    print(f"FPS: {loader.fps}")
    print(f"Total frames to analyze: {len(loader)}")
    
    print("Starting analyzing:")
    
    for i, (timestamp, img) in enumerate(loader):
        print(f"Klatka {i}: Czas {timestamp:.2f}s | Rozmiar obrazka: {img.size}")
        
        if i == 0:
            img.show()
            
        if i >=2:
            break
        
except Exception as e:
    print(f"Error occured: {e}")
    