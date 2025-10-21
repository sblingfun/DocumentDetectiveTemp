
from ollama import chat, ChatResponse
from pydantic import BaseModel

import vector_db
import embedder


class ResponseRank(BaseModel):
    vec_id: int
    score: float
    text: str

def form_rerank_query(query: str, text: str, vector_id: int):
    #return f"Rate the relevance of the following sections of text to the query being asked, only returning a float value between 1 and 10. Consider the specific context and intent of the query, not just keyword matches. Only return the rating, no other text. Query: {query}, Text Section: {text}, Relevance Score:"
    return f"Rate the relevance of the following sections of text to the query being asked as a float value between 1 and 10. Consider the specific context and intent of the query, not just keyword matches. Query: {query}, Text Section: {text}, Vector ID: {vector_id}, Only respond with valid JSON where vec_id is the Vector ID, text is the Text Section, and score is the relevancy score generated."


def rerank_results(query: str, num_results: int):
    
    initial_results = vector_db.query_vector_db(embedder.convert_query_embedding(query), num_results)
    #initial_results = vector_db.query_vector_db(embedder.convert_query_embedding(query),3)
    result_text_list = []
    print(initial_results)
    for result in initial_results:
        ranking_response: ChatResponse = chat(
            model="llama3.1:8b",
            messages=[
                {
                    "role": "user",
                    "content": form_rerank_query(query, result.payload["text"], result.id),
                }
            ],
            format=ResponseRank.model_json_schema(),
        )
        result_text_list.append(ResponseRank.model_validate_json(ranking_response.message.content))
        #print(ranking_response.message.content)
    print(result_text_list)
    result_text_list.sort(key=lambda rank: rank.score, reverse=True)
    print(result_text_list)


