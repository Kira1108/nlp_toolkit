import os
import numpy as np
from typing import List
from .chroma_adapter import make_chroma


class OpenAIEmbedding:
    def __init__(self, model:str = 'text-embedding-ada-002'):
        import openai
        from openai import OpenAI
        openai.api_key = os.environ['OPENAI_API_KEY']
        self.client = OpenAI()
        self.model = model
        
    def make_chroma(self):
        return make_chroma(self)
        
    def __call__(self, texts:List[str]):
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        response = self.client.embeddings.create(input = texts, model=self.model)
        return [embd.embedding for embd in response.data]
