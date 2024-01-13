import os
from typing import List, Dict
from uuid import uuid4
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Distance, VectorParams, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


qdrant_client = QdrantClient(
    url=os.environ["QDRANT_URL"], 
    api_key=os.environ["QDRANT_API_KEY"]
)

try:
    qdrant_client.get_collection(collection_name="chats")
except Exception as e:
    qdrant_client.create_collection(
        collection_name="chats",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    
def generate_embeddings(data: List[str]):
    return model.encode(data)

def generate_embeddings(data: str):
    return model.encode([data])[0].tolist()

def store_embeddings(vector: List[float], payload: Dict[str, any]):
    operation_info = qdrant_client.upsert(
        collection_name="chats",
        wait=True,
        points=[
            PointStruct(id=str(uuid4()), vector=vector, payload=payload),
        ],
    )
    return operation_info

def similarity_search(query_vector: List[float], limit: int = 3, must_filters: List = None, score_threshold: float = 0.8):
    search_result = qdrant_client.search(
        collection_name="chats", 
        query_vector=query_vector, 
        limit=limit, 
        query_filter=Filter(
            must=[
                FieldCondition(
                    key=filter[0],
                    match=MatchValue(value=filter[1]),
                ) for filter in must_filters
            ]
        ),
        score_threshold=score_threshold
    )
    return search_result

def similarity_scroll(must_filters: List = None):
    scroll_result = qdrant_client.scroll(
        collection_name="chats", 
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key=filter[0],
                    match=MatchValue(value=filter[1]),
                ) for filter in must_filters
            ]
        ),
        with_payload=True,
        with_vectors=False
    )[0]
    return scroll_result

def update_indexed_status():
    qdrant_client.set_payload(
        collection_name="chats",
        payload={
            "indexed": True,
        },
        points=Filter(
            must=[
                FieldCondition(
                    key="indexed",
                    match=MatchValue(value=False),
                ),
            ],
        ),
    )
    