import pandas as pd
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import Dataset
import os
import argparse

def generate_prompts_retrieval(documents, question):
    prompt_template = f"""
    #Role
    You are an experienced expert skilled in answering various questions.

    #Task
    Please answer the question based on the documents provided and following the detailed requirements using the format '#Answer:'

    #Reference Documents
    {documents}

    #Requirements
    Please consider the retrieved documents provided '#Reference Documents' and answer the question.

    #Question
    {question}
    """
    return prompt_template

def prepare_training_data(df, target_column):
    data = []
    for index, row in df.iterrows():
        question = row['question']
        documents = "\n".join([row[f'doc{i}'] for i in range(1, 6)])  # Join all 5 documents
        prompt = generate_prompts_retrieval(documents, question)
        
        # Add the generated prompt and corresponding response to the dataset
        data.append({
            'prompt': prompt,
            'completion': row.get(target_column, '')  # Use the target column for response
        })
    
    return pd.DataFrame(data)

def tokenize_function(examples, tokenizer):
    return tokenizer(examples['prompt'], text_target=examples['completion'], truncation=True, padding='max_length')

def main(args):
    df = pd.read_csv(args.data_path)
    train_data = prepare_training_data(df, args.target_column)
    dataset = Dataset.from_pandas(train_data)

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")  #change with meta-llama/Llama-2-13b-hf
    tokenized_datasets = dataset.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)

    
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf") #change with meta-llama/Llama-2-13b-hf
    
    
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=3,
        weight_decay=0.001,
        save_total_limit=2,
        save_strategy="epoch",
        logging_dir='./logs',  
        logging_steps=10,
        report_to="none",  
    )
    
    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        tokenizer=tokenizer
    )
    
    # Fine-tune the model
    trainer.train()
    
    # Save the model and tokenizer
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning Llama-2s")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the input CSV file")
    parser.add_argument('--target_column', type=str, required=True, help="column name ('output_crag' or 'target')")
    parser.add_argument('--output_dir', type=str, default="./fine_tuned_model", help="Directory to save the model")
    
    args = parser.parse_args()
    
    main(args)
