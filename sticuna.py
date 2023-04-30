from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("stable-vicuna-13b")
model = AutoModelForCausalLM.from_pretrained("stable-vicuna-13b")
model.half().cuda()

prompt = """\
### Human: Write a Python script for text classification using Transformers and PyTorch
### Assistant:\
"""

inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
tokens = model.generate(
 **inputs,
 max_new_tokens=256,
 do_sample=True,
 temperature=1.0,
 top_p=1.0,
)
print(tokenizer.decode(tokens[0], skip_special_tokens=True))
