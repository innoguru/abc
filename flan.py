from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl")

def ask_question(question):
    input_prompt = f"{question}"
    input_ids = tokenizer.encode(input_prompt, return_tensors="pt")
    output_ids = model.generate(input_ids)
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return answer

question = "What is the second amendment?"

answer = ask_question(question)
print(answer)