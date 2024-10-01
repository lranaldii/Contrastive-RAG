import argparse
import torch
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

def download_wikipedia_dump():
    wiki_dataset = load_dataset("wikipedia", "20220301.en", split="train")
    return wiki_dataset

def load_dpr_model():
    # Context Encoder
    context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

    # Question Encoder
    question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

    return context_encoder, context_tokenizer, question_encoder, question_tokenizer

def load_ms_marco_model():
    # Model and tokenizer for MS MARCO
    model = AutoModel.from_pretrained("microsoft/msmarco-bert-base-dot-v5")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/msmarco-bert-base-dot-v5")

    return model, tokenizer

def encode_wikipedia_documents(wiki_dataset, context_encoder, context_tokenizer):
    document_embeddings = []
    for doc in wiki_dataset:
        inputs = context_tokenizer(doc['text'], return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            embedding = context_encoder(**inputs).pooler_output
        document_embeddings.append(embedding)

    return torch.cat(document_embeddings)

def retrieve_similar_documents(query, question_encoder, question_tokenizer, document_embeddings, wiki_dataset):
    inputs = question_tokenizer(query, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        query_embedding = question_encoder(**inputs).pooler_output

    similarity_scores = cosine_similarity(query_embedding.detach().numpy(), document_embeddings.detach().numpy())
    top_indices = np.argsort(similarity_scores[0])[-5:]  # Top 5 documents

    top_documents = [wiki_dataset[i]['text'] for i in top_indices]
    return top_documents

def process_csv_and_retrieve_documents(csv_path, model_type, output_csv):
    print("Downloading Wikipedia dump...")
    wiki_dataset = download_wikipedia_dump()

    if model_type == 'DPR':
        print("Loading DPR models...")
        context_encoder, context_tokenizer, question_encoder, question_tokenizer = load_dpr_model()
    elif model_type == 'MS-MARCO':
        print("Loading MS MARCO models...")
        context_encoder, context_tokenizer = load_ms_marco_model()
        question_encoder, question_tokenizer = context_encoder, context_tokenizer  # MS MARCO doesn't distinguish

    print("Encoding Wikipedia documents...")
    document_embeddings = encode_wikipedia_documents(wiki_dataset, context_encoder, context_tokenizer)

    df = pd.read_csv(csv_path)

    # Step 5: Retrieve top 5 documents for each question
    results = []
    for index, row in df.iterrows():
        question = row['question']
        target = row['target']
        top_documents = retrieve_similar_documents(question, question_encoder, question_tokenizer, document_embeddings, wiki_dataset)

        # Store the question, retrieved documents, and target
        results.append({
            'question': question,
            'doc1': top_documents[0] if len(top_documents) > 0 else "",
            'doc2': top_documents[1] if len(top_documents) > 1 else "",
            'doc3': top_documents[2] if len(top_documents) > 2 else "",
            'doc4': top_documents[3] if len(top_documents) > 3 else "",
            'doc5': top_documents[4] if len(top_documents) > 4 else "",
            'target': target
        })

    output_df = pd.DataFrame(results)
    output_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

def main(args):
    process_csv_and_retrieve_documents(args.input_csv, args.model_type, args.output_csv)

if __name__ == "__main__":
    # Setup argparse to handle command-line arguments
    parser = argparse.ArgumentParser(description="Retrieval using DPR MS-MARCO and save the results to CSV.")
    parser.add_argument("--model_type", type=str, choices=['DPR', 'MS-MARCO'], required=True, help="Choose the retrieval model: 'DPR' or 'MS-MARCO'")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV file with 'question' and 'target' columns")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save the output CSV file with questions, documents, and target")

    args = parser.parse_args()
    main(args)
