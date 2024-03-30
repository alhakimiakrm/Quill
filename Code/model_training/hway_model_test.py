from transformers import GPT2LMHeadModel, GPT2Tokenizer

def text_gen(prompt, hwaymodel='./results', length=50):
    tokenizer = GPT2Tokenizer.from_pretrained(hwaymodel)
    model = GPT2LMHeadModel.from_pretrained(hwaymodel)

    encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

    output = model.generate(
        input_ids=encoded_prompt,
        max_length=length + len(encoded_prompt[0]),
        temperature=0.7,  # Adjusted temperature for more varied output
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2,
        do_sample=True,
    )

    generated_sequence = output[0].tolist()
    text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
    text = text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)):]

    return text.strip()

# Example usage
prompt = "The old man washed away the sins from his past."
generated_text = text_gen(prompt)
print(prompt, generated_text)