from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter


client = QdrantClient(url="http://localhost:6333")
collection_name = "nutrition_collection"


def create_vector_db_collection(vector_size: int):
    client.create_collection(collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
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

def collect_vector_context(vector_idx: int, num_neighbors: int):
    #need to reference chunekr overlap size
    #possibly need to include payload
    #need to check for minimum and maximum values in range
    context = client.retrieve(collection_name=collection_name,
                           ids=list(range(vector_idx - num_neighbors, vector_idx + num_neighbors + 1)))
    context_text = ""
    for item in context:
        context_text = context_text + item.payload["text"]
    return context_text

#This might not work properly with qdrants vector indexing
def num_vectors():
    count = client.count(collection_name=collection_name)
    print("Count obj:")
    print(count)
    return count["count"]
