# Vector Database Workflow

### 1. Set proxy if using in ....
```python
import pandas as pd
from nlp_toolkit import (
    set_proxy,
    SentenceEmbedder,
    make_chroma,
    insert_df_collection
)

set_proxy()
```


### 2. Set up embedder
```python
sentences = description.app_desc.tolist()
embedder = SentenceEmbedder(mps = True, batch_size=256)

# this is the object used by chromadb
chroma_embedder = make_chroma(embedder)
chroma_embedder(['what is the fuck']).shape
```


### 3. Create embeddings
```python
embeded_sentences = embedder(sentences)
```

### 4. Insert to vector database
```python
import chromadb

chroma_client = chromadb.PersistentClient(path='./chroma_storage')

collection = chroma_client.create_collection(name="app", embedding_function=chroma_embedder)

insert_df_collection(
    collection,
    embeded_sentences, 
    description, 
    'app_desc', 
    'id', 
    meta_cols = ['track_name']
)
```


### 5. Query vector database
```python
results = collection.query(
    query_texts=["music music score, instrument, guitar, piano, bass, band, sound, tunning, scale,"],
    n_results=20,
)

results['metadatas']
```

```bash
Embedding Batch: 1......Done.
[[{'track_name': 'TonalEnergy Chromatic Tuner and Metronome'},
  {'track_name': 'iReal Pro - Music Book & Play Along'},
  {'track_name': 'Easy Music - Give kids an ear for music'},
  {'track_name': 'PrestoBand Guitar and Piano'},
  {'track_name': 'Piano - Play Keyboard Music Games with Magic Tiles'},
  {'track_name': 'ABRSM Aural Trainer Grades 1-5'},
  {'track_name': 'Magic Piano by Smule'},
  {'track_name': 'Free Piano app by Yokee'},
  {'track_name': 'SOUND Canvas'},
  {'track_name': 'Mastering the piano with Lang Lang'},
  {'track_name': 'Final Guitar - absolute guitar app'},
  {'track_name': 'Guitar Suite - Metronome, Tuner, and Chords Library for Guitar, Bass, Ukulele'},
  {'track_name': 'Tongo Music - for kids and families'},
  {'track_name': 'QQ音乐-来这里“发现・音乐”'},
  {'track_name': 'OnSong'},
  {'track_name': 'Music Memos'},
  {'track_name': 'Cytus'},
  {'track_name': 'Musicloud Pro - MP3 and FLAC Music Player for Cloud Platforms.'},
  {'track_name': 'Musicloud - MP3 and FLAC Music Player for Cloud Platforms.'},
  {'track_name': 'Epic Orchestra'}]]
```