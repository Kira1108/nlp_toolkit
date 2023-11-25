import os
import numpy as np
from typing import List


class OpenAIEmbedding:
    def __init__(self, model:str = 'text-embedding-ada-002'):
        import openai
        from openai import OpenAI
        openai.api_key = os.environ['OPENAI_API_KEY']
        self.client = OpenAI()
        self.model = model
        
    def __call__(self, texts:List[str]):
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        response = self.client.embeddings.create(input = texts, model=self.model)
        return [embd.embedding for embd in response.data]
