import asyncio
import os
from typing import Optional, List
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from .utils import logger

class BGEEmbeddingLocal:
    """Local BGE embedding model wrapper"""
    
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or self._get_best_device()
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _get_best_device(self) -> str:
        """Determine the best available device"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _load_model(self):
        """Load the tokenizer and model"""
        logger.info(f"Loading BGE model {self.model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info("BGE model loaded successfully!")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((RuntimeError, torch.cuda.CudaError)),
    )
    async def encode(self, texts: List[str], model: Optional[str] = None) -> np.ndarray:
        """Main encode method that handles batching"""
        try:
            print(f"Getting embedding for {len(texts)} chunks")
            
            # Tokenize - matching the original approach
            encoded = self.tokenizer(
                texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512  # BGE max length    
            ).to(self.device)
        
            with torch.no_grad():
                outputs = self.model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"]
                )

                # Use simple mean of last hidden state
                embeddings = outputs.last_hidden_state.mean(dim=1)
        
            # Handle dtype conversion matching original
            if embeddings.dtype == torch.bfloat16:
                result = embeddings.detach().to(torch.float32).cpu().numpy()
            else:
                result = embeddings.detach().cpu().numpy()
            
            print(f"Received embedding response: {len(embeddings)} embeddings")
            return result
        
        except Exception as e:
            logger.exception("An error occurred during BGE embedding.")
            raise e

# Global singleton storage
_bge_model_instances = {}

def get_bge_model_instance(model_name: str = "BAAI/bge-large-en-v1.5") -> BGEEmbeddingLocal:
    global _bge_model_instances
    if model_name not in _bge_model_instances:
        _bge_model_instances[model_name] = BGEEmbeddingLocal(model_name)
    return _bge_model_instances[model_name]
        
        