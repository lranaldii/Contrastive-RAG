import pandas as pd
import openai
import argparse

# Load the CSV file containing documents and questions
def load_data(file_path):
    return pd.read_csv(file_path)

# Generate prompts for the CRAG mode
def generate_prompts(documents, question):
    prompt_template = f"""
    #Role
    You are an experienced expert skilled in answering various questions.

    #Task
    Please answer the question based on the documents provided and following the detailed requirements.

    #Reference Documents
    {documents}

    #Requirements
    1) Please consider the retrieved documents provided “#Reference Documents” and understand the main points. 
    2) For each document, after extracting the most helpful passages discuss whether they are actually relevant or irrelevant for answering the #Question.
    3) Please consider the passages in step 2) in detail, ensure they are correct. 
    4) Finally, extract the answer in a short and concise format by marking it as “#Answer:”.

    #Question
    {question}
    """
    return prompt_template

def generate_prompts_baseline(question):
    prompt_template = f"""
    #Role
    You are an experienced expert skilled in answering various questions.

    #Task
    Please answer the question following the detailed requirements.

    #Requirements
    Please answer the question based on your knowledge using the format “#Answer:” 

    #Question
    {question}
    """
    return prompt_template

def generate_prompts_retrieval(documents, question):
    prompt_template = f"""
    #Role
    You are an experienced expert skilled in answering various questions.

    #Task
    Please answer the question based on the documents provided and following the detailed requirements using the format “#Answer:”

    #Reference Documents
    {documents}

    #Requirements
    Please consider the retrieved documents provided “#Reference Documents” and answer the question.

    #Question
    {question}
    """
    return prompt_template

def call_gpt4_api(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4o-2024-05-13",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

def process_csv_and_query_api(file_path, mode):
    data = load_data(file_path)
    
    # Prepare a new DataFrame to store outputs
    outputs = pd.DataFrame()

    for index, row in data.iterrows():
        documents = "\n".join([f"[{i+1}] {row[f'doc{i+1}']}" for i in range(5)])  # Adjusted for 5 documents
        question = row['question']
        
        if mode == 'baseline':
            prompt = generate_prompts_baseline(question)
            answer = call_gpt4_api(prompt)
            row['output_baseline'] = answer
        elif mode == 'retrieval':
            prompt = generate_prompts_retrieval(documents, question)
            answer = call_gpt4_api(prompt)
            row['output_retrieval'] = answer
        elif mode == 'crag':
            prompt = generate_prompts(documents, question)
            answer = call_gpt4_api(prompt)
            row['output_crag'] = answer
        
        # Append the row to the outputs DataFrame
        outputs = outputs.append(row, ignore_index=True)

    # Save the output DataFrame to a new CSV file
    outputs.to_csv('output_with_answers.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Query GPT-4 with specified mode.')
    parser.add_argument('file', type=str, help='Path to the CSV file containing documents and questions')
    parser.add_argument('--mode', choices=['baseline', 'retrieval', 'crag'], required=True, 
                        help='Mode to use for querying: baseline, retrieval, or crag')
    args = parser.parse_args()

    process_csv_and_query_api(args.file, args.mode)

  #python query_gpt.py your_file.csv --mode crag / retrieval / baseline
