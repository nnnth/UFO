import argparse
import sys 
sys.path.append('.')
import torch
from internvl.model.internvl_chat import InternVLChatModel
from transformers import AutoTokenizer

argparse = argparse.ArgumentParser()
argparse.add_argument('input_path', type=str, help='Path to the input model')
argparse.add_argument('output_path', type=str, help='Path to the output model')
argparse.add_argument('lora_path', type=str, help='Path to the lora weight')
args = argparse.parse_args()

print('Loading model...')
model = InternVLChatModel.from_pretrained(
    args.input_path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16).eval()
print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(args.input_path, trust_remote_code=True)
# Add <MASK>
tokenizer.add_tokens(['<MASK>'])
num_new_tokens = 1
model.language_model.resize_token_embeddings(len(tokenizer))
output_embeddings = model.language_model.get_output_embeddings().weight.data
output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
output_embeddings[-num_new_tokens:] = output_embeddings_avg

model.config.llm_config.vocab_size = len(tokenizer)
model.language_model.config.vocab_size = len(tokenizer)

# wrap lora
model.wrap_llm_lora(r=8, lora_alpha=2 * 8)
lora_state = torch.load(args.lora_path)
new_state = {}
for key, value in lora_state.items():
    newkey = key.split("backbone.")[1]
    print(newkey)
    new_state[newkey] = value 

model.load_state_dict(new_state, strict=False)
model.language_model.merge_and_unload()
model.language_model = model.language_model.model
model.config.use_llm_lora = 0

print('Saving model...')
model.save_pretrained(args.output_path)
print('Saving tokenizer...')
tokenizer.save_pretrained(args.output_path)
print('Done!')
