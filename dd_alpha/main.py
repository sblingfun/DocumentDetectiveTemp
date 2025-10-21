import embedder
import llm_manager

import os
import pandas as pd
import numpy as np
import torch

import vector_db
import llm_prompt_generator
#this import should be removed to seperate class
from sentence_transformers import util
import textwrap
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available
from transformers import BitsAndBytesConfig

from huggingface_hub import login

login()

temp_csv_file = "temp_chunks_and_embeddings_df.csv"
if (os.path.exists(temp_csv_file)):
    print("path exists")
else:
    print("Path DNE")

#text_chunks_and_embedding_df = embedder.convert_csv_to_tensor_embeddings("temp_chunks_and_embeddings_df.csv")
#chunks_and_tensors = pd.read_csv("temp_chunks_and_embeddings_df.csv")
#text_chunks_and_embedding_df = pd.read_csv("temp_chunks_and_embeddings_df.csv")

#Should be moved to external function
#text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
#pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")

#embedding_tensor = embedder.convert_embeddings_to_tensor(text_chunks_and_embedding_df)
#embedding_tensor = torch.tensor(np.array(text_chunks_and_embedding_df["embedding"].tolist()), dtype=torch.float32).to("cuda")

#print(embedding_tensor.shape)
#print(text_chunks_and_embedding_df.head())

current_query = ""
#Move query function into seperate class

current_query = input("Embedding Data Loaded, please enter a query: \n")

query_embedding = embedder.convert_text_embedding(current_query)

print("Querying vector db")

query_results = vector_db.query_vector_db(query_embedding, 5)
print("Query results")
print(query_results)
query_context_strings = []
for item in query_results:
    print(item.payload.get('text'))
    query_context_strings.append(item.payload.get('text'))
    
print("Query String results")
print(query_context_strings)

prompt_with_context = llm_prompt_generator.prompt_formatter(current_query, query_context_strings)

print("prompt w context: ")
print(prompt_with_context)





#dot_scores = util.dot_score(a=query_embedding, b=embedding_tensor)[0]
#dot_scores = embedder.query_dot_product(current_query, embedding_tensor)

#top_results_dot_product = torch.topk(dot_scores, k=5)
#print(top_results_dot_product)

def print_wrapped(text, wrap_length=60):
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)

'''
for score, idx in zip(top_results_dot_product[0], top_results_dot_product[1]):
    print(f"Score: {score:.4f}")
    print("Text:")
    print_wrapped(pages_and_chunks[idx]["sentence_chunk"])
    print(f"Page number: {pages_and_chunks[idx]['page_number']}")
    print("\n")
'''
#gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
#gpu_memory_gb = round(gpu_memory_bytes / (2**30))
#print(f"Available GPU memory: {gpu_memory_gb} GB")

#model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
model_id = "google/gemma-2-2b-it"

quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

if (is_flash_attn_2_available()) and (torch.cuda.get_device_capability(0)[0] >= 8):
  attn_implementation = "flash_attention_2"
else:
  attn_implementation = "sdpa"
print(f"[INFO] Using attention implementation: {attn_implementation}")

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id)
#llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_id,torch_dtype=torch.float16,low_cpu_mem_usage=False)
#llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_id,attn_implementation=attn_implementation,low_cpu_mem_usage=False)
#llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_id,quantization_config=quantization_config,attn_implementation=attn_implementation,low_cpu_mem_usage=False)
#llm_model.to("cuda")

llm_model_class = llm_manager.Llm("google/gemma-2-2b-it")

llm_model = llm_manager.get_llm_model(llm_model_class)


#LLM info
print(llm_model)
def get_model_num_params(model: torch.nn.Module):
    return sum([param.numel() for param in model.parameters()])

get_model_num_params(llm_model)
def get_model_mem_size(model: torch.nn.Module):
    """
    Get how much memory a PyTorch model takes up.

    See: https://discuss.pytorch.org/t/gpu-memory-that-model-uses/56822
    """
    # Get model parameters and buffer sizes
    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    mem_buffers = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])

    # Calculate various model sizes
    model_mem_bytes = mem_params + mem_buffers # in bytes
    model_mem_mb = model_mem_bytes / (1024**2) # in megabytes
    model_mem_gb = model_mem_bytes / (1024**3) # in gigabytes

    return {"model_mem_bytes": model_mem_bytes,
            "model_mem_mb": round(model_mem_mb, 2),
            "model_mem_gb": round(model_mem_gb, 2)}

get_model_mem_size(llm_model)

def get_model_mem_size(model: torch.nn.Module):
    """
    Get how much memory a PyTorch model takes up.

    See: https://discuss.pytorch.org/t/gpu-memory-that-model-uses/56822
    """
    # Get model parameters and buffer sizes
    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    mem_buffers = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])

    # Calculate various model sizes
    model_mem_bytes = mem_params + mem_buffers # in bytes
    model_mem_mb = model_mem_bytes / (1024**2) # in megabytes
    model_mem_gb = model_mem_bytes / (1024**3) # in gigabytes

    return {"model_mem_bytes": model_mem_bytes,
            "model_mem_mb": round(model_mem_mb, 2),
            "model_mem_gb": round(model_mem_gb, 2)}

print(get_model_mem_size(llm_model))


input_text = "What are the macronutrients, and what roles do they play in the human body?"
#print(f"Input text:\n{input_text}")

# Create prompt template for instruction-tuned model
dialogue_template = [
    {"role": "user",
     "content": current_query}
]

# Apply the chat template
'''
prompt = tokenizer.apply_chat_template(conversation=dialogue_template,
                                       tokenize=False, # keep as raw text (not tokenized)
                                       add_generation_prompt=True)
'''
#print(f"\nPrompt (formatted):\n{prompt}")

input_ids = tokenizer(prompt_with_context, return_tensors="pt").to("cuda")
print(f"Model input (tokenized):\n{input_ids}\n")

# Generate outputs passed on the tokenized input
# See generate docs: https://huggingface.co/docs/transformers/v4.38.2/en/main_classes/text_generation#transformers.GenerationConfig 
outputs = llm_model.generate(**input_ids,
                             max_new_tokens=600) # define the maximum number of new tokens to create
print(f"Model output (tokens):\n{outputs[0]}\n")

outputs_decoded = tokenizer.decode(outputs[0])
print(f"Model output (decoded):\n{outputs_decoded}\n")

print("Batch Decode")
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))



