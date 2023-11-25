import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
from .chroma_adapter import make_chroma

@dataclass
class SentenceEmbedder:

    model_ckpt:str = "sentence-transformers/all-MiniLM-L6-v2"
    mps:bool = False
    batch_size:int = 512

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_ckpt)
        self.model = AutoModel.from_pretrained(self.model_ckpt)
        if self.mps:
            self.device ="mps"
        else:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
    def make_chroma(self):
        return make_chroma(self)

    def _embed(self, sentences):
        encoder_input = self.tokenizer(sentences, padding = True, truncation = True, return_tensors = 'pt').to(self.device)

        with torch.no_grad():
            model_output = self.model(**encoder_input)

        return encoder_input, model_output

    def mean_pooling(self, encoder_input, model_output):

        attention_mask = encoder_input['attention_mask']

        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sentence_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min = 1e-9)
        sentence_embeddings = F.normalize(sentence_embeddings, p = 2, dim = 1)
        return sentence_embeddings

    def embed_sentences(self, sentences):
        encoder_input, model_output = self._embed(sentences)
        return self.mean_pooling(encoder_input, model_output).cpu().numpy()

    def __call__(self, texts):
        batch_size = self.batch_size
        n_batchs = len(texts) // batch_size + int(len(texts) % batch_size > 0)

        subset_embeddings = []

        for i in tqdm(range(n_batchs), total = n_batchs, desc = "Embedding Batches"):
            subset = texts[i * batch_size: (i+1) * batch_size]
            subset_embedding = self.embed_sentences(subset)
            subset_embeddings.append(subset_embedding)
        return np.concatenate(subset_embeddings)
  


        
    
    
    
    


    


