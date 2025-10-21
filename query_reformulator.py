


#include temp 0
def increase_query_precision(query: str):
    precision_prompt = f"You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system. Given the user query, reformulate it to be more percise, specific, detailed, and likely to return relrevant results. Write it to be more specific, detailed, and likely to retrieve relevant information. Only return the reformulated query, no other text. User Query: {query}"
    return precision_prompt


def extract_query_keywords(query: str):
    keyword_prompt = f"You are an AI assistant tasked with extracting 3-8 key terms or phrases that best capture the essence of a user query. Focus on nouns, key concepts, and any domain-specific terminology. Given the user query, extract the key search terms from the query and return them as a list separating terms with commas. User Query: {query}"
    return keyword_prompt

def decompose_query(query: str):
    decomposition_prompt = f"You are an AI assistant tasked with breaking down complex queries into simplier sub-queries for a RAG system. Given the original query, decompose it into 2-4 simpler sub-queries, that when answered together would provide a comprehensive response to the original query. Only return the list of sub-queries, no other text. Original query: {query} /n Use the following example to guide your response: Example: What are the impacts of climate change on the environment? /n Sub-queries: /n 1.What are the impacts of climate change on biodiversity? /n 2. How does climate change affect the oceans? /n What are the effects of climate change on agriculture? /n What are the impacts of climate change on human health?"
    return decomposition_prompt
