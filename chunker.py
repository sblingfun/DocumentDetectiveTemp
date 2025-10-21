from spacy.lang.en import English
from tqdm.auto import tqdm
import re
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

nlp = English()

nlp.add_pipe("sentencizer")

class Chunker:
    def __init__(self, num_sentence_chunk_size, chunk_overlap_size):
        self.num_sentence_chunk_size = num_sentence_chunk_size
        self.chunk_overlap_size = chunk_overlap_size

def sentence_via_spacy(pages_and_texts: list):
    for item in tqdm(pages_and_texts):
        item["sentences"] = list(nlp(item["text"]).sents)

        item["sentences"] = [str(sentence) for sentence in item["sentences"]]

        item["page_sentence_count_spacy"] = len(item["sentences"])

def split_list(input_list: list,
        slice_size: int) -> list[list[str]]:

    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]


def set_sentence_chunk_size(input_chunk_size: int):
    num_sentence_chunk_size = input("Enter desired sentence chunk size")
    return

def chunk_input(pages_and_texts: list, num_sentence_chunk_size: int):
    for item in tqdm(pages_and_texts):
        item["sentence_chunks"] = split_list(input_list=item["sentences"], slice_size=num_sentence_chunk_size)
        item["num_chunks"] = len(item["sentence_chunks"])


def chunk_input_via_lang(pages_and_texts: list):
   text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200, length_function=len)
   for item in tqdm(pages_and_texts):
        item["sentence_chunks"] = text_splitter.split_text(item["text"])


#should include definition for chunk item dict

def create_chunk_data_item(pages_and_texts: list) -> list:
    pages_and_chunks = []
    for item in tqdm(pages_and_texts):
    	for sentence_chunk in item["sentence_chunks"]:
    	    chunk_dict = {}
    	    chunk_dict["page_number"] = item["page_number"]
    	    
    	    #joining chunk sentencens in paragraph
    	    joined_sentence_chunk = "".join(sentence_chunk).replace("  "," ").strip()
    	    joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk)
    	    chunk_dict["sentence_chunk"] = joined_sentence_chunk
    	    
    	    chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
    	    chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
    	    #need to keep token size as global var
    	    chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4
    	    
    	    pages_and_chunks.append(chunk_dict)

    return pages_and_chunks

def filter_by_min_token_size(pages_and_chunks: list, min_token_length: int):
    for item in pages_and_chunks:
        if item["chunk_token_count"] <= min_token_length:
            pages_and_chunks.remove(item)
            
def chunk_via_langchain(pdf_path):
    document = UnstructuredPDFLoader(pdf_path).load()
    #need to be able to set chunk size dynamically
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=200, length_function=len)
    texts = text_splitter.split_documents(document)
    #cleaned_texts = replace_t_with_space(texts)
    
    return texts
    
    


