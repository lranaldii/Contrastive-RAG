import pandas as pd
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Function to generate the retrieval-based prompt
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

def perform_inference(df, model, tokenizer, device):
    # Storage for generated answers
    results = []

    for index, row in df.iterrows():
        # Prepare the input prompt
        question = row['question']
        documents = "\n".join([row[f'doc{i}'] for i in range(1, 6)]) 
        prompt = generate_prompts_retrieval(documents, question)

        # Tokenize the input prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate model output with specified parameters
        outputs = model.generate(
            inputs['input_ids'], 
            max_length=2048,  # Maximum generation length
            temperature=0.4,  # Temperature for sampling
            num_return_sequences=1,
            do_sample=True
        )

        # Decode the output tokens and extract the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Store the generated response
        results.append({
            'question': question,
            'generated_answer': generated_text
        })

    return pd.DataFrame(results)

def load_dataset(test_file):
    return pd.read_csv(test_file)

def load_model_and_tokenizer(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def setup_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description="Perform inference using a fine-tuned Llama-2s.")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the test CSV file.")
    parser.add_argument("--model_path", type=str, default='./fine_tuned_model', help="Path to the fine-tuned model.")
    parser.add_argument("--output_file", type=str, default='inferences.csv', help="Path to save the output CSV file.")
    args = parser.parse_args()

    df_test = load_dataset(args.test_file)

    model, tokenizer = load_model_and_tokenizer(args.model_path)

    device = setup_device()
    model.to(device)

    inference_results = perform_inference(df_test, model, tokenizer, device)

    inference_results.to_csv(args.output_file, index=False)

    print(f"Inference completed. Results saved to {args.output_file}.")
