import json
from pathlib import Path
import chromadb
from chromadb.config import Settings
from embeddings.embedder import Embedder
from config import ROOT, CHROMA_PATH,COLLECTION

def ingest():
    chunks=json.loads((ROOT/'documents.json').read_text(encoding='utf-8'))
    encoder=Embedder()
    client=chromadb.PersistentClient(path=CHROMA_PATH,settings=Settings(anonymized_telemetry=False))
    col=client.get_or_create_collection(COLLECTION,metadata={'hnsw:space':'cosine'},embedding_function=None)
    col.upsert(ids=[str(i) for i in range(len(chunks))],documents=[x['text'] for x in chunks],metadatas=[{'source':x['source'],'section':x['section']} for x in chunks],embeddings=encoder.embed_batch([x['text'] for x in chunks]))
    assert col.count()==6
    print('ChromaDB: 6 документов загружены и сохранены на диск.')
if __name__=='__main__': ingest()
