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
    collection = collection,
    embeddings = embeded_sentences, 
    df = dataframe, 
    doc_col = 'app_desc', 
    id_col = 'id', 
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


### 6. Connect to existing collection & making queries.
```python
import chromadb
import pandas as pd
from nlp_toolkit import (
    set_proxy,
    SentenceEmbedder,
    make_chroma
)

set_proxy()

embedder = SentenceEmbedder(mps = True, batch_size=256)

chroma_embedder = make_chroma(embedder)

chroma_client = chromadb.PersistentClient(path='./chroma_storage')

collection = chroma_client.get_collection(name="app", embedding_function=chroma_embedder)

results = collection.query(
    query_texts=["monitor your sleep, keep healthy life style"],
    n_results=20,
)

results['metadatas']
```

```bash
Using http proxy: http://127.0.0.1:8001
Turn on VPN with corresponding proxy.
Embedding Batch: 1......Done.
[[{'track_name': 'iSleeping by iSommeil SARL'},
  {'track_name': 'Good Morning Alarm Clock - Sleep Cycle Tracker'},
  {'track_name': 'Sleep Cycle alarm clock'},
  {'track_name': 'Sleep Pulse 2 Motion - The Sleep Tracker for Watch'},
  {'track_name': 'SenseSleep - Train Your Brain To Sleep Better'},
  {'track_name': 'SmartFit - Wristband'},
  {'track_name': 'Sleep Meister - Sleep Cycle Alarm'},
  {'track_name': 'AutoSleep. Auto Sleep Tracker for Watch'},
  {'track_name': 'Sleep Meister - Sleep Cycle Alarm Lite'},
  {'track_name': 'Smart Alarm Clock : sleep cycle & snoring recorder'},
  {'track_name': 'Life Cycle - Track Your Time Automatically'},
  {'track_name': 'Sleep Talk Recorder'},
  {'track_name': 'Morning Routine : Daily Habit Tracker'},
  {'track_name': 'airweave sleep analysis'},
  {'track_name': 'Snail Sleep-Dream Talk Recording'},
  {'track_name': 'Productive habits & daily goals tracker'},
  {'track_name': 'iSleep Easy - Meditations for Restful Sleep'},
  {'track_name': 'Fitness Sync for Fitbit to Apple Health'},
  {'track_name': 'SnoreLab : Record Your Snoring'},
  {'track_name': 'Kiwake Alarm Clock - Take back your mornings'}]]
```