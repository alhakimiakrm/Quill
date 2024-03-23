from transformers import GPT2LMHeadModel, GPT2Tokenizer

#This is a script to test GPT2's functionality as a pre-trained model that we could pass our data through

def generate_text(prompt_text, model_name='gpt2-medium', length=100):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    
    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
    output_sequences = model.generate(input_ids=encoded_prompt, max_length=length, temperature=1.0)

    generated_sequence = output_sequences[0].tolist()
    text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

    text = text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
    
    return text.strip()


prompt = "There are never any suicides in the quarter among people one knows," 
generated_text = generate_text(prompt)
print(prompt, generated_text)
