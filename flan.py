from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

def ask_question(question, context):
    input_prompt = f"question: {question} context: {context}"
    input_ids = tokenizer.encode(input_prompt, return_tensors="pt")
    output_ids = model.generate(input_ids)
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return answer

context = "OpenAI is an artificial intelligence research lab consisting of the for-profit corporation OpenAI LP and its parent company, the non-profit OpenAI Inc."
question = "Who founded OpenAI?"

answer = ask_question(question, context)
print(answer)
