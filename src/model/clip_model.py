import torch
from transformers import CLIPProcessor, CLIPModel

class CLIPEngine:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device = None):
        
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        
        else:
            self.device = device
        
        print(f"Chosen device: {self.device}")
        
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        
    def get_image_embeddings(self, images):
        vector_inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.get_image_features(**vector_inputs)
        
        outputs = outputs / outputs.norm(p=2, dim = -1, keepdim=True) #p=2 -> Euclidesan Norm
        return outputs.cpu().numpy()
    
    def get_text_embeddings(self, text_list):
        vector_inputs = self.processor(text=text_list, return_tensors="pt", padding = True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.get_text_features(**vector_inputs)
        
        outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
        return outputs.cpu().numpy()
        