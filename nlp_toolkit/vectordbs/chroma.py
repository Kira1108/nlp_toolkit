import numpy as np
import datetime
import pandas as pd
from typing import List

def insert_df_collection(collection, 
                         embeddings:List[List[float]],
                         df:pd.DataFrame, 
                         doc_col:str, 
                         id_col:str, 
                         meta_cols:list = None) -> None:
    
    df['create_time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df['update_time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    ids = df[id_col].astype(str).tolist()

    if isinstance(embeddings, np.ndarray):
        embeddings = embeddings.tolist()

    documents = df[doc_col].tolist()
    
    if meta_cols is None:
        meta_cols = [col for col in df.columns if col not in [id_col, doc_col]]
    
    metadatas = df[meta_cols].to_dict(orient = 'records')
    
    params = dict(
        documents = documents,
        embeddings = embeddings,
        metadatas = metadatas,
        ids = ids
    )
    
    params = {k:v for k,v in params.items() if v is not None}
    
    collection.add(**params)
    
    print("Successful added to collection")
    
    
def insert_df_collection_batch(
    collection, 
    embeddings:List[List[float]], 
    df:pd.DataFrame, 
    doc_col:str, 
    id_col:str = None, 
    meta_cols:list = None, 
    batch_size = 10000) -> None:
    
    n_batchs = int(np.ceil(len(df)/batch_size))
    
    for i in range(n_batchs):
        print(f"Inserting batch {i+1}/{n_batchs}")
        insert_df_collection(
            collection, 
            embeddings[i*batch_size:(i+1)*batch_size], 
            df.iloc[i*batch_size:(i+1)*batch_size], 
            doc_col, 
            id_col, 
            meta_cols)
        
    print("Successful added all data to collection")
    
    
def update_metafield(collection, ids = None, **kwargs):
    """The only thing you can update is the metafield."""
    
    # if no ids is provided, update all ids.
    if ids is None:
        ids = collection.get(include=[])['ids']
      
    # add an updatetime to update meta dictionary  
    kwargs.update(update_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    collection.update(
        ids = ids,
        metadatas = [kwargs] * len(ids)
    )
    
    print("Update metdata successfully.")
    
    
def mean_seed_query(collection, ids, n:int = 10):

   embeddings = collection.get(ids = ids, include = ['embeddings'])['embeddings']

   mean_vector = np.array(embeddings).mean(axis = 0)

   results = collection.query(
      query_embeddings = [mean_vector.tolist()], 
      n_results = n)
   
   return results

def concat_text_query(collection, ids, n:int = 10):
    
   documents = collection.get(ids = ids, include = ['documents'])['documents']

   query_text = " ".join(documents)

   results = collection.query(
         query_texts = [query_text], 
         n_results = n)
   
   return results
    

    


