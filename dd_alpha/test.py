import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain import hub
from langchain_core.documents import Document
from typing_extensions import List, TypedDict

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

llm = ChatOllama(
        model="llama3.2:1b",
        temperature=0,
        )

embeddings = OllamaEmbeddings(model="llama3")
vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",
        )

bs4_strainer = bs4.SoupStrainer(class_=("post-title","post-header", "post-content"))

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
    )

docs = loader.load()

assert len(docs) == 1
print(f"Total characters: {len(docs[0].page_content)}")

text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
        )
all_splits = text_splitter.split_documents(docs)

print(f"Split blog post into {len(all_splits)} sub-documents.")

document_ids = vector_store.add_documents(documents=all_splits)

print(document_ids[:3])

prompt = hub.pull("rlm/rag-prompt")

example_messages = prompt.invoke({"context" : "(context here)", "question": "(question here)"}).to_messages()

assert len(example_messages) == 1
print(example_messages[0].content)

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

var = input("Ask a question: ")
inputState = State()
inputState["question"] = var
print(inputState["question"])


#inputState["context"] = retrieve(inputState)
#retrieve(inputState)
#if isinstance(inputState["context"], List[Document]):
    #print("List of Documents")

#if isinstance(inputState["context"], str):
    #print("String")

#print(type(inputState["context"]))

inputState["context"] = vector_store.similarity_search(inputState["question"])

print(inputState["question"])
print(inputState["context"])

inputState["answer"] = generate(inputState)
#print(generate(inputState))


print(inputState["question"])
print(inputState["answer"])
