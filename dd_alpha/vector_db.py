from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


client = QdrantClient(url="http://localhost:6333")
collection_name = "temp_collection"


def create_vector_db_collection(vector_size: int):
    client.create_collection(collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.DOT)
    )


def insert_into_vector_db(input_vector):
    #print("input vec: ", input_vector)
    return client.upsert(collection_name=collection_name,
            points=[input_vector],
            )

def query_vector_db(query_vector, result_size: int):
    #print("query vec: ", query_vector)
    return client.query_points(collection_name=collection_name,
        query=query_vector,
        with_payload=True,
        limit=result_size
    ).points
    
def delete_collection():
    client.delete_collection(collection_name=collection_name)
    
def does_collection_exist() -> bool:
    return client.collection_exists(collection_name=collection_name)


