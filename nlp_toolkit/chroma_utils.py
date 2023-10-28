import numpy as np

def insert_df_collection(collection, 
                         embeddings,
                         df, doc_col:str, 
                         id_col:str = None, 
                         meta_cols:list = None):

    if id_col is not None:
        ids = df[id_col].astype(str).tolist()
    else:
        ids = None
        id_col = ""
        
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