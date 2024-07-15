import torch
import transformers

from utils import constants

model_id = "meta-llama/Llama-2-7b-chat-hf"
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load the model and configuration
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True,
    access_token=constants.hf_token,
)
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id, access_token=constants.hf_token
)

model.eval()

tokenizer.save_pretrained("efs_mount\model")
model.save_pretrained("efs_mount\model")
