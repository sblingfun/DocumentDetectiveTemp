from sentence_transformers import util, SentenceTransformer
from tqdm.auto import tqdm
import pandas as pd
import torch
import numpy as np
from qdrant_client.models import PointStruct
import vector_db

embedding_model = Sentencenembedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", 
                                      device="cuda") # choose the device to load the model to (note: GPU will often be *much* faster than CPU)
                                      
embeddings_df_save_path = "temp_chunks_and_embeddings_df.csv"

#device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda"

#Will want to include storage for vector db


def combine_chunks(pages_and_chunks: list) -> list:
    return [item["sentence_chunk"] for item in pages_and_chunks]

#Should include batch size and possible store as tensor
#text_chunk_embeddings = embedding_model.encode(text_chunks, batch_size=32, convert_to_tensor=True)

def create_embeddings(pages_and_chunks: list):
    for item in tqdm(pages_and_chunks):
        item["embedding"] = embedding_model.encode(item["sentence_chunk"])
        #can pull embedding size to create vector db with correct size from length of embedding
        #print("embedding size: " , len(item["embedding"]))
    return pages_and_chunks

def save_embeddings_to_file(pages_and_chunks: list):
    df_to_save = pd.DataFrame(pages_and_chunks)
    df_to_save.to_csv(embeddings_df_save_path, index=False)
    
def set_embeddings_df_save_path(save_path: str):
    embeddings_df_save_path = save_path
    
def convert_csv_to_tensor_embeddings(csv_filename: str):
    chunks_and_embeddings_df = pd.read_csv(csv_filename)
    #print(chunks_and_embeddings_df.head())
    #print(chunks_and_embeddings_df["embedding"].dtype)
    #chunks_and_embeddings_df["embedding"] = chunks_and_embeddings_df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
    return chunks_and_embeddings_df
    
def convert_embeddings_to_tensor(chunks_and_embeddings: list):
    #Clearn excess df creation
    #temp_df = pd.DataFrame(chunks_and_embeddings)
    return torch.tensor(np.array(chunks_and_embeddings["embedding"].tolist()), dtype=torch.float32).to(device)
    
def convert_text_embedding(text_to_embed: str):
    #return embedding_model.encode(text_to_embed, convert_to_tensor=True)
    return embedding_model.encode(text_to_embed)
    
def query_dot_product(query_to_compare: str, embedding_tensor):
    return util.dot_score(a=convert_text_embedding(query_to_compare),b=embedding_tensor)[0]
    
def create_and_store_embeddings(pages_and_chunks: list):
    #shouldnt delete if trying to store data
    #if vector_db.does_collection_exist:
        #vector_db.delete_collection
    #vector_db.delete_collection
    vector_db.create_vector_db_collection(768)
    idx = 0 
    for item in tqdm(pages_and_chunks):
        idx += 1       
        vector_db.insert_into_vector_db(PointStruct(id=idx,vector=embedding_model.encode(item["sentence_chunk"]), payload = {
        "text": item["sentence_chunk"]
        },))
    
    
