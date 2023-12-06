from .base import BaseEmbedder
import numpy as np

def make_chroma(embedder:BaseEmbedder):
    from chromadb import Documents, EmbeddingFunction, Embeddings

    class ChromaEmbeddingFunction(EmbeddingFunction):
        def __call__(self, input: Documents) -> Embeddings:
            result = embedder(input)
            if isinstance(result, list):
                return result
            else:
                return result.tolist()
        
    return ChromaEmbeddingFunction()