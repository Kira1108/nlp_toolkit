from .base import BaseEmbedder

def make_chroma(embedder:BaseEmbedder):
    from chromadb import Documents, EmbeddingFunction, Embeddings

    class ChromaEmbeddingFunction(EmbeddingFunction):
        def __call__(self, input: Documents) -> Embeddings:
            return embedder(input)
        
    return ChromaEmbeddingFunction()