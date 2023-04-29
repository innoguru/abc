from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xxl")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xxl")

def ask_question(question, context):
    input_prompt = f"{question}"
    input_ids = tokenizer.encode(input_prompt, return_tensors="pt")
    output_ids = model.generate(input_ids)
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return answer

context = ""
question = "What is the second amendment and when was it written? Write a little essay about it."

answer = ask_question(question, context)
print(answer)
