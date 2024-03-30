import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Function for preprocessing the dataset
def preprocess_dataset(file_path):
    # Implement any necessary preprocessing steps here
    # For example, removing special characters, handling newlines, etc.
    pass

def train_gpt2():
    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    model = GPT2LMHeadModel.from_pretrained('gpt2-medium')

    # Paths to your training and test data
    train_path = '../Hemingway/hemingway1.txt'
    test_path = '../Hemingway/hemingway1.txt'

    # Preprocess datasets
    preprocess_dataset(train_path)
    preprocess_dataset(test_path)

    # Prepare dataset
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_path,
        block_size=512)  # Increased block size

    test_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=test_path,
        block_size=512)  # Increased block size

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        eval_steps=400,
        save_steps=800,
        warmup_steps=500,
        learning_rate=5e-5,  # Adjusted learning rate
        gradient_accumulation_steps=8,  # Added for handling larger effective batch sizes
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    # Start training
    trainer.train()
    model.save_pretrained('./results')
    tokenizer.save_pretrained('./results')

if __name__ == '__main__':
    train_gpt2()