import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.utils import is_flash_attn_2_available

class Llm:
    def __init__(self, model_id):
        self.model_id = model_id

#should try to use singleton to ensure only 1 llm active at once
#can add enum and stiwtch statement on initialization to allow for multiple llm models
def get_llm_model(inputmodel):
    #model_id = "google/gemma-2-2b-it"
    model_id = inputmodel.model_id
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    if (is_flash_attn_2_available()) and (torch.cuda.get_device_capability(0)[0] >= 8):
        attn_implementation = "flash_attention_2"
    else:
        attn_implementation = "sdpa"

    llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_id,quantization_config=quantization_config,attn_implementation=attn_implementation,low_cpu_mem_usage=False)
    llm_model.to("cuda")

    return llm_model

