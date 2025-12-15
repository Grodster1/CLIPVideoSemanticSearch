import torch
from src.model.clip_model import CLIPEngine
import numpy as np
import json
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


class SearchEngine:
    def __init__(self, index_dir):
        self.model = CLIPEngine()
        self.index_dir = index_dir
        self.vectors = np.load(os.path.join(index_dir, 'vectors.npy'))
        
        with open(os.path.join(index_dir, 'metadata.json'), 'r') as f:
            self.metadata = json.load(f)
            
    
    def search(self, qeuery, top_n=5):
        tex_emb=self.model.get_text_embeddings([qeuery])
        scores = (self.vectors @ tex_emb.T).squeeze()
        best_indices = np.argsort(scores)[::-1][:top_n]
        
        results = []
        
        for idx in best_indices:
            results.append({
                'timestamp':self.metadata[idx],
                'score':float(scores[idx])
            })
        
        return results
            
                