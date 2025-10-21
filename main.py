from ollama import chat, ChatResponse

import embedder
import vector_db
import llm_prompt_generator
import query_reformulator
import reranker

while True:

    current_query = input("Please enter a query: \n")

    reranker.rerank_results(current_query, 5)
'''
    response: ChatResponse = chat(model="llama3.1:8b", messages=[
                                  {
                                    "role": "user",
                                    "content": query_reformulator.increase_query_precision(current_query),
                                  },])

    print(response.message.content)

    response: ChatResponse = chat(model="llama3.1:8b", messages=[
                                  {
                                    "role": "user",
                                    "content": query_reformulator.extract_query_keywords(current_query),
                                  },])

    print(response.message.content)


    response: ChatResponse = chat(model="llama3.1:8b", messages=[
                                  {
                                    "role": "user",
                                    "content": query_reformulator.decompose_query(current_query),
                                  },])

    print(response.message.content)

    num = vector_db.num_vectors
    print("num " , num)
    original = vector_db.collect_vector_context(60,0)
    results = vector_db.collect_vector_context(60,2)

    print(original)
    print("context")
    print(results)


    result = vector_db.query_vector_db(embedder.convert_query_embedding(current_query),5)

    for item in result:
        print(item)
    
    llm_prompt = llm_prompt_generator.prompt_formatter(current_query, [item.payload["text"] for item in result]) 
    print("LLM Prompt")
    print(llm_prompt)

    response: ChatResponse = chat(model="llama3.1:8b", messages=[
                                  {
                                    "role": "user",
                                    "content": llm_prompt,
                                  },])

    print(response.message.content)
'''
