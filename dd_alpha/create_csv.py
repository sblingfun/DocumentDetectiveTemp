import uploadpdf as pdf
import chunker
import embedder
import pandas as pd

#exec("/home/alex/Documents/personal/ragLLM/uploadpdf.py")

pdf.Pdf_path = pdf.get_pdf_path_name()

print("Converting pdf into digital text")
#pages_and_texts = pdf.open_and_read_pdf(pdf.Pdf_path.pdf_path)

#pages_and_texts = pdf.open_and_read_pdf_ocr(pdf.Pdf_path.pdf_path)
#pages_and_texts = pdf.open_and_read_pdf_ocr_old(pdf.Pdf_path.pdf_path)

#print(pages_and_texts[1115])

#pages_and_texts = chunker.chunk_via_langchain(pdf.Pdf_path.pdf_path)

pages_and_texts = pdf.open_and_ocr_pdf(pdf.Pdf_path.pdf_path)

df = pd.DataFrame(pages_and_texts)
print(df.head())


#chunking_func = chunker.Chunker(10)

print("Chunking text")
#chunker.sentence_via_spacy(pages_and_texts)

chunker.chunk_input(pages_and_texts, 10)

#print(pages_and_texts[1])
#print(pages_and_texts[1115])

pages_and_chunks = chunker.create_chunk_data_item(pages_and_texts)

#print(len(pages_and_chunks))
#print(pages_and_chunks[1298])

#df = pd.DataFrame(pages_and_chunks)
#print(df.describe().round(2))

chunker.filter_by_min_token_size(pages_and_chunks, 30)

#df = pd.DataFrame(pages_and_chunks)
#print(df.describe().round(2))
#print(df[:2])

print("Creating Embeddings")
#chunks_and_embeddings = embedder.create_embeddings(pages_and_chunks)
embedder.create_and_store_embeddings(pages_and_chunks)

#df = pd.DataFrame(chunks_and_embeddings)
#print(df.head())

#This process of saving to file is for demo purposes will be storing in vector db
#embedder.save_embeddings_to_file(chunks_and_embeddings)


#embedding_tensor = embedder.convert_embeddings_to_tensor(chunks_and_embeddings)

#print(embedding_tensor.shape)

print("Done")

