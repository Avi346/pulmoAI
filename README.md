# AI Radiologist Assistant

A simple AI-powered tool that classifies chest X-rays as **Normal** or **Pneumonia** using a fine-tuned ResNet-18 model, and retrieves relevant clinical guidelines using a RAG (Retrieval-Augmented Generation) pipeline.

## Project Structure

```
project3/
├── data/
│   ├── chest_xray/          # Kaggle dataset
│   └── guidelines/           # Medical guideline PDFs
├── models/                   # Saved model weights
├── vectorstore/              # ChromaDB vector store
├── src/
│   ├── train_vision.py       # Vision pipeline (training + prediction)
│   └── build_rag.py          # RAG pipeline (indexing + querying)
├── app.py                    # Streamlit frontend application
├── requirements.txt
└── README.md
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the Dataset (Kaggle API)

```bash
pip install kaggle
# Place your kaggle.json in C:\Users\<YourUser>\.kaggle\
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p data/
# Unzip (PowerShell)
Expand-Archive -Path data\chest-xray-pneumonia.zip -DestinationPath data\
```

### 3. Add Medical PDFs

Download 2-3 pneumonia guideline PDFs and place them in `data/guidelines/`:
- [WHO Childhood Pneumonia Guidelines (2014)](https://apps.who.int/iris/bitstream/handle/10665/137319/9789241507813_eng.pdf)
- [ATS/IDSA CAP Guidelines (2019)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6812437/) — download PDF from page

### 4. Configure Mistral AI

Copy `.env.example` to `.env` and set `MISTRAL_API_KEY` to your Mistral API key. You can also set `MISTRAL_API_KEY` in your environment. The `.env` file is ignored by Git.

### 5. Build the Guideline Index

```bash
python src/build_rag.py --build
```

The chest X-ray dataset, generated vector store, temporary uploads, and caches stay local and are excluded from Git. Model weights and guideline PDFs are included in the repository.

## Usage

### Train the Vision Model

```bash
python src/train_vision.py
```

Trains DenseNet-121 (384x384 resolution) with layer-wise discriminative fine-tuning, focal loss, and automated Otsu lung cropping. Saves weights to `models/chest_xray_densenet121.pth`.

### Run the Streamlit Frontend

```bash
streamlit run app.py
```

Launches an interactive web dashboard with tabs for X-ray classification (with TTA) and conversational RAG clinical consulting.

## Performance Benchmark (Test Set Evaluation)

Evaluated on 624 unseen test chest radiographs (`data/chest_xray/test`):

| Class | Precision | Recall (Sensitivity) | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **NORMAL** | `0.97` | `0.67` | `0.79` | 234 |
| **PNEUMONIA** | `0.83` | **`0.99`** | **`0.90`** | 390 |
| **Overall Accuracy** | — | — | **`87%`** | 624 |

> [!TIP]
> **Clinical Screening Priority**: In diagnostic radiograph screening, **Sensitivity (Pneumonia Recall of 99%)** is paramount. The network caught **385 out of 390** active pneumonia cases (only 5 false negatives), ensuring patient safety.

## Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| Vision Model | PyTorch, DenseNet-121 (384x384 aspect-padding + TTA) |
| Preprocessing | Scikit-Image (Otsu Lung Cropping) |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) |
| Vector Store | ChromaDB |
| RAG Framework | LangChain |
| LLM | Mistral AI API |
