from .base import BaseEmbedder

def make_chroma(embedder:BaseEmbedder):
    from chromadb import Documents, EmbeddingFunction, Embeddings

    class ChromaEmbeddingFunction(EmbeddingFunction):
        def __call__(self, texts: Documents) -> Embeddings:
            return embedder(texts)
        
    return ChromaEmbeddingFunction()