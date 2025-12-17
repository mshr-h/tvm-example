from transformers import AutoConfig, AutoTokenizer
import onnxruntime
import numpy as np
from huggingface_hub import hf_hub_download

# 1. Setup
## Load config and tokenizer
model_id = "onnx-community/granite-4.0-h-350m-ONNX"
config = AutoConfig.from_pretrained(model_id) 
tokenizer = AutoTokenizer.from_pretrained(model_id)

## Load model
filename = "model" # Options: "model", "model_q4", "model_fp16", "model_q4f16"
dtype = np.float32 # or np.float16 if using fp16/q4f16
model_path = hf_hub_download(model_id, subfolder="onnx", filename=f"{filename}.onnx") # Download Graph
hf_hub_download(model_id, subfolder="onnx", filename=f"{filename}.onnx_data") # Download Weights
decoder_session = onnxruntime.InferenceSession(model_path)
output_names = [o.name for o in decoder_session.get_outputs()]

## Initialize config values
num_key_value_heads = config.num_key_value_heads
head_dim = config.hidden_size // config.num_attention_heads
eos_token_id = config.eos_token_id
d_conv = config.mamba_d_conv
mamba_n_heads = config.mamba_n_heads
mamba_d_head = config.mamba_d_head
mamba_d_state = config.mamba_d_state
mamba_n_groups = config.mamba_n_groups
mamba_expand = config.mamba_expand
hidden_size = config.hidden_size
conv_d_inner = (mamba_expand * hidden_size) + (2 * mamba_n_groups * mamba_d_state)

# 2. Prepare inputs
## Define messages
messages = [
  { "role": "user", "content": "What is the capital of France?" },
]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="np")
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']
batch_size = input_ids.shape[0]

## Initialize cache
cache = {}
for i, layer_type in enumerate(config.layer_types):
  if layer_type == "attention":
    for kv in ('key', 'value'):
      cache[f'past_key_values.{i}.{kv}'] = np.zeros([batch_size, num_key_value_heads, 0, head_dim], dtype=dtype)
  elif layer_type == "mamba":
    cache[f'past_conv.{i}'] = np.zeros([batch_size, conv_d_inner, d_conv], dtype=dtype)
    cache[f'past_ssm.{i}'] = np.zeros([batch_size, mamba_n_heads, mamba_d_head, mamba_d_state], dtype=dtype)

# 3. Generation loop
max_new_tokens = 1024
generated_tokens = np.array([[]], dtype=np.int64)
for i in range(max_new_tokens):
  feed_dict = dict(
    input_ids=input_ids,
    attention_mask=attention_mask,
  )
  outputs = decoder_session.run(None, feed_dict | cache)
  named_outputs = dict(zip(output_names, outputs))

  ## Update values for next generation loop
  input_ids = outputs[0][:, -1].argmax(-1, keepdims=True)
  attention_mask = np.concatenate([attention_mask, np.ones_like(input_ids, dtype=np.int64)], axis=-1)
  
  for name in cache:
    new_name = name.replace('past_key_values', 'present').replace('past_', 'present_')
    cache[name] = named_outputs[new_name]

  generated_tokens = np.concatenate([generated_tokens, input_ids], axis=-1)
  if (input_ids == eos_token_id).all():
    break

  ## (Optional) Streaming
  print(tokenizer.decode(input_ids[0]), end='', flush=True)
print()

# 4. Output result
print(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0])
