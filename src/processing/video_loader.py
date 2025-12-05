import cv2
from PIL import Image
import os


class VideoLoader:
    def __init__(self, video_path, sample_per_n_seconds = 2):
        if not os.path.exists(video_path):
            raise FileExistsError(f"Video file not found: {video_path}")
        
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open a video: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.duration = self.total_frames/self.fps
        
        self.frame_hop = int(self.fps * sample_per_n_seconds)
        if self.frame_hop < 1:
            self.frame_hop = 1
            
    def __len__(self):
        return int(self.total_frames // self.frame_hop)
    
    def __iter__(self):
        current_frame = 0
        
        while current_frame < self.total_frames:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            read, frame = self.cap.read()
            if not read:
                break
            
            frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_RGB)
            timestamp = current_frame / self.fps
            
            yield timestamp, pil_image
            
            current_frame += self.frame_hop
        
        self.cap.release()