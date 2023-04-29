from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl")

def ask_question(question, num_answers=3):
    input_prompt = f"{question}"
    input_ids = tokenizer.encode(input_prompt, return_tensors="pt")
    output_ids = model.generate(input_ids, num_return_sequences=num_answers)
    
    answers = []
    for idx in range(num_answers):
        answer = tokenizer.decode(output_ids[idx], skip_special_tokens=True)
        answers.append(answer)
    return answers

question = "What is the second amendment and when was it written? Write a little essay about it."

answers = ask_question(question)
for i, answer in enumerate(answers, start=1):
    print(f"Answer {i}: {answer}\n")
