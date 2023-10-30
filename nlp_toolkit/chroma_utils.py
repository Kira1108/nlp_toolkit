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
    
    
def insert_df_collection_batch(collection, embeddings, df, doc_col:str, id_col:str = None, meta_cols:list = None, batch_size = 10000):
    
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