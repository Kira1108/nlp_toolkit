from collections import Counter
from typing import List
import string
import numpy as np
from dataclasses import dataclass
import os
from pathlib import Path
import json
import pickle
import nltk
from nltk.corpus import stopwords


def remove_stop_words(input_string:str):
    nltk.download('stopwords')
    """
    This function removes stop words from the given text.
    """
    stop_words = set(stopwords.words('english'))
    words = input_string.split()
    filtered_words = [word for word in words if word.casefold() not in stop_words]
    return ' '.join(filtered_words)

def remove_punctuation_and_numbers(input_string:str):
    translator = str.maketrans('', '', string.punctuation + string.digits)
    no_punct = input_string.translate(translator)
    return no_punct

def make_lowercase(input_string:str):
    return input_string.lower()

class Tokenizer:
    
    def tokenize_doc(self, doc:str) -> list:
        """Tokenize a single document of string type"""
        if not isinstance(doc, str):
            raise ValueError("doc must be a string")
        doc = remove_punctuation_and_numbers(doc)
        doc = make_lowercase(doc)
        doc = remove_stop_words(doc)
        return [word.strip() for word in doc.lower().split(" ")]
        
    def tokenize_batch(self, documents:List[str]) -> List[List[str]]:
        """Tokenize a batch of documents of string type"""
        if not isinstance(documents, list):
            raise ValueError("documents must be a list of strings")
        return [self.tokenize_doc(doc) for doc in documents]
    
    def tokenize(self, documents:List[str] | str) -> List[List[str]]:
        """Tokenize either a batch of documents or a single document"""
        if isinstance(documents, list):
            return self.tokenize_batch(documents)
        
        elif isinstance(documents, str):
            return [self.tokenize_doc(documents)]
        
        else:
            raise ValueError("Input Error")
        
    def __call__(self, doc):
        """Tokenizer entry point"""
        return self.tokenize(doc)


@dataclass
class BM25Model:
    
    k1:float
    b:float
    tokenizer:Tokenizer = Tokenizer()
    fitted:bool = False
    
    def dump(self, path:str = "bm25_model_dump"):
        if not self.fitted:
            raise ValueError("Model not fitted yet")
        
        dump_folder = Path(path)
        
        if not os.path.exists(dump_folder):
            os.makedirs(dump_folder)
            
        tokenizer_path = dump_folder / "tokenizer.pkl"
        idf_dict_path = dump_folder / "idf_dict.json"
        document_word_counts_path = dump_folder / "document_word_counts.pkl"
        documents_path = dump_folder / "documents.pkl"
        params_path = dump_folder / "params.json"
        params = {"k1":self.k1, "b":self.b, "avg_doc_len":self.avg_doc_len}
        
        # save parameters as a json file
        with open(params_path, "w") as f:
            json.dump(params, f)
        
        # save ifdf data to a json file
        with open(idf_dict_path, "w") as f:
            json.dump(self.idf_dict, f)
            
        # save tokenizer to a pickle
        with open(tokenizer_path, "wb") as f:
            pickle.dump(self.tokenizer,f)
        
        # save document_word_counts to a pickle
        with open(document_word_counts_path, "wb") as f:
            pickle.dump(self.document_word_counts,f)
        
        # save documents to a pickle
        with open(documents_path, "wb") as f:
            pickle.dump(self.documents,f)
    
    def tokenize_documents(self, documents:List[str]) -> List[List[str]]:
                
        self.documents = documents
        
        self.tokenized_documents = self.tokenizer(documents)
        
        self.document_word_counts = [
            dict(Counter(tokenized_doc)) 
            for tokenized_doc in self.tokenized_documents
        ]
        
        self.avg_doc_len = np.mean([len(doc) for doc in self.tokenized_documents])
    
    @classmethod  
    def load(cls, path):
        load_path = Path(path)
        
        tokenizer_path = load_path / "tokenizer.pkl"
        idf_dict_path = load_path / "idf_dict.json"
        document_word_counts_path = load_path / "document_word_counts.pkl"
        documents_path = load_path / "documents.pkl"
        params_path = load_path / "params.json"
        
        # load tokenizer
        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)
            
        # load idf_dict
        with open(idf_dict_path, "r") as f:
            idf_dict = json.load(f)
            
        # load document_word_counts
        with open(document_word_counts_path, "rb") as f:
            document_word_counts = pickle.load(f)
            
        # load documents
        with open(documents_path, "rb") as f:
            documents = pickle.load(f)
            
        # load params
        with open(params_path, "r") as f:
            params = json.load(f)
            
        obj = cls(k1 = params['k1'], b = params['b'], tokenizer = tokenizer, fitted = True)
        obj.idf_dict = idf_dict
        obj.document_word_counts = document_word_counts
        obj.documents = documents
        obj.avg_doc_len = params['avg_doc_len']
        
        return obj
            
        
    def compute_idf(self, tokenized_documents: List[List[str]]) -> dict[str, float]:
        N = len(tokenized_documents)
        idf_dict = {}
        for tokenized_doc in tokenized_documents:
            for word in set(tokenized_doc):
                idf_dict[word] = idf_dict.get(word, 0) + 1       

        for word, doc_count in idf_dict.items():
            idf_dict[word] = np.log((N - doc_count + 0.5) / (doc_count + 0.5) + 1)
        
        self.idf_dict = idf_dict
        
    def fit(self, documents):
        self.tokenize_documents(documents)        
        self.compute_idf(self.tokenized_documents)
        self.fitted = True
        return self
    
    def get_score(self, query, doc):
        score = 0
        # tokenized query
        tokenized_query = self.tokenizer.tokenize_doc(query)
        # tokenize doc, with counter
        doc_word_count = dict(Counter(self.tokenizer.tokenize_doc(doc)))
        # for each word in query
        for word in tokenized_query:
            # check the occurance of the word in the doc
            word_cnt = doc_word_count.get(word, 0)
            
            score += self.idf_dict.get(word,0) * (
                word_cnt * (self.k1 + 1) / (word_cnt + self.k1 * (1 - self.b + self.b * len(doc) / self.avg_doc_len)) 
                )
        return score
    
    def search(self, query:str,n:int = None):
        
        tokenized_query = self.tokenizer.tokenize_doc(query)
        
        scores = []
        for doc_wc in self.document_word_counts:
            score = 0
            for word in tokenized_query:
                word_cnt = doc_wc.get(word, 0)
                score += self.idf_dict.get(word,0) * (
                    word_cnt * (self.k1 + 1) / (word_cnt + self.k1 * (1 - self.b + self.b * len(doc_wc) / self.avg_doc_len)) 
                    )
            scores.append(score)

        idx = np.argsort(scores)[::-1]
        
        if n is None:
        
            return {
                "scores": np.array(scores)[idx].tolist(),
                "documents": np.array(self.documents)[idx].tolist()
            }
        else:
            return {
                "scores": np.array(scores)[idx].tolist()[:n],
                "documents": np.array(self.documents)[idx].tolist()[:n]
            }