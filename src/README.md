
# Contrastive RAG pipeline

This repository contains a series of scripts for reproducing our C-RAG experiments.

Below are the steps on how to run each script, with detailed explanations of their functionality.


## CRAG

This script reads a CSV file with documents and questions and uses a selected mode to perform specific tasks. 

### Usage:

```bash
python crag.py your_file.csv --mode retrieval
```

### Arguments:
- `task.csv`: Path to the CSV file containing questions and documents of selected tasks.
- `--mode`: Selects the mode in which the script will run. Options:
  - `retrieval`: Use retrieval-augmented generation.
  - `baseline`: Use the baseline method.
  - `crag`: Use the CRAG (Contrastive Retrieval Augmented Generation) method.

### Example:

```bash
python crag.py your_file.csv --mode retrieval
```

### Functionality:
- **Retrieval Mode**: This mode retrieves relevant documents based on the question and produces prompts for further processing.
- **Baseline Mode**: A simpler processing model that generates answers based on predefined rules.
- **CRAG Mode**: A contrastive retrieval model designed to refine the retrieved results using contrastive learning.

---

## tuning

This script fine-tunes a Llama-2-7b/13b using the data provided in a CSV file. 

### Usage:

```bash
python tuning.py --data_path path/annotated_file.csv --target_column output_crag --output_dir ./fine_tuned_model
```

### Arguments:
- `--data_path`: Path to the CSV file containing training data.
- `--target_column`: The column in the CSV file that contains the desired output for fine-tuning. (e.g., `output_crag`)
- `--output_dir`: The directory where the fine-tuned model will be saved.

### Example:

```bash
python tuning.py --data_path ./task_file.csv --target_column output_crag --output_dir ./fine_tuned_model
```

### Functionality:
- **Model Fine-Tuning**: Fine-tunes the pre-trained model using the question-document pairs in the CSV file.
- **Target Column**: The `target_column` specifies which column in the CSV file contains the outputs you want the model to learn.
- **Model Saving**: Once the training is complete, the fine-tuned model will be saved in the directory specified by `--output_dir`.

---

## `test/inference

This script runs inference on a fine-tuned model using a test CSV file. It generates the model's predictions based on the questions and documents provided in the CSV file.

### Usage:

```bash
python test.py --test_file ./test_csv.csv
```

### Arguments:
- `--test_file`: Path to the CSV file containing test data (questions and documents).

### Example:

```bash
python test.py --test_file ./test_csv.csv
```

### Functionality:
- **Inference**: Uses the fine-tuned model to generate answers based on the test CSV file.
- **CSV Output**: The generated answers are saved in a CSV file.

---

## Retriever (construction dataframes)

This script performs document retrieval using either the DPR (Dense Passage Retrieval) or MS-MARCO model. It takes an input CSV file containing questions and retrieves the top 5 documents for each question. The results are saved in a CSV file.

### Usage:

```bash
python retriever.py --model_type DPR --input_csv questions.csv --output_csv results.csv
```

### Arguments:
- `--model_type`: Selects the retrieval model type (`DPR` or `MS-MARCO`).
- `--input_csv`: Path to the input CSV file containing the `question` and `target` columns.
- `--output_csv`: Path to the output CSV file where the retrieved documents will be saved.

### Example:

```bash
python retriever.py --model_type DPR --input_csv questions.csv --output_csv results.csv
```

### Functionality:
- **Document Retrieval**: Retrieves the top 5 most similar documents for each question based on the chosen model (either `DPR` or `MS-MARCO`).
- **CSV Output**: The retrieved documents are saved in the specified CSV file along with the original questions and targets.

---

## Requirements

The following Python packages are required to run the scripts:

- `transformers`
- `datasets`
- `torch`
- `pandas`
- `scikit-learn`
- `numpy`
- `argparse`

---

## NB

- **Data building-step**: The document retrieval task requires a large memory footprint due to the storage of document embeddings. Consider adjusting the dataset size and batch size based on your hardware.

