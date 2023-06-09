from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xxl")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xxl")

print("Model loaded to VRAM.")

def ask_question(question):
    input_prompt = f"{question}"
    input_ids = tokenizer.encode(input_prompt, return_tensors="pt")
    output_ids = model.generate(input_ids, max_length=512, min_length=200)  # Adjust max_length and min_length
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return answer

question = "What is the second amendment and when was it written? Write a little essay about it."

answer = ask_question(question)
print(answer)