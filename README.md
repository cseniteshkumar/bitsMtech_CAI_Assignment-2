# Conversational AI Assignment 2 - Hybrid RAG System

**Group ID: 45**

## Team Members

| Name                | Student ID    | Contribution |
|---------------------|---------------|--------------|
| VAIBHAV SAREEN      | 2024AA05923   | 100%         |
| LAWLESH KUMAR       | 2024AA05149   | 100%         |
| VIVEK TRIVEDI       | 2024AA05922   | 100%         |
| NITESH KUMAR        | 2024AA05143   | 100%         |
| LOGESH M            | 2024AA05163   | 100%         |

## Project Overview

This project implements a **Hybrid Retrieval-Augmented Generation (RAG) system** that combines:
- **Dense Vector Retrieval** using sentence embeddings (FAISS)
- **Sparse Keyword Retrieval** using BM25 algorithm
- **Reciprocal Rank Fusion (RRF)** to combine results from both methods
- **LLM-based Answer Generation** using transformers

The system answers questions from a corpus of 500 Wikipedia articles (200 fixed + 300 random) and includes an automated evaluation framework with 100 generated questions.

## System Architecture

The system consists of the following components:

1. **Data Collection**: Wikipedia article fetching and preprocessing
2. **Text Chunking**: 200-400 token chunks with 50-token overlap
3. **Dual Retrieval System**:
   - Dense retrieval using sentence-transformers
   - Sparse retrieval using BM25
4. **Reciprocal Rank Fusion**: Combines results from both retrievers
5. **Answer Generation**: Uses transformer-based LLM (Flan-T5)
6. **Automated Evaluation**: Generates questions and evaluates performance

## Dependencies

### Python Version
- Python 3.8 or higher

### Required Libraries

```bash
# Core dependencies
wikipedia-api>=0.6.0
beautifulsoup4>=4.12.0
requests>=2.32.0

# NLP and Embeddings
sentence-transformers>=2.2.0
transformers>=4.41.0
nltk>=3.8.0

# Vector Search and Ranking
faiss-cpu>=1.7.4
rank-bm25>=0.2.2

# Deep Learning
torch>=2.0.0

# Scientific Computing
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0

# Utilities
tqdm>=4.65.0
```

## Installation Steps

### 1. Clone the Repository

```bash
git clone <repository-url>
cd bitsMtech_CAI_Assignment-2
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install all required packages
pip install wikipedia-api beautifulsoup4
pip install sentence-transformers faiss-cpu
pip install rank-bm25 nltk
pip install transformers torch
pip install tqdm numpy scipy scikit-learn
```

### 4. Download NLTK Data

Open Python and run:

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
```

## Run Instructions

### System Execution

The complete system can be run using the Jupyter Notebook:

#### Option 1: Using Jupyter Notebook

```bash
# Install Jupyter if not already installed
pip install jupyter

# Launch Jupyter Notebook
jupyter notebook

# Open: Conversational_AI_Assignment_2_Gorup_45.ipynb
# Run all cells sequentially
```

#### Option 2: Using JupyterLab

```bash
# Install JupyterLab if not already installed
pip install jupyterlab

# Launch JupyterLab
jupyter lab

# Open and run: Conversational_AI_Assignment_2_Gorup_45.ipynb
```

#### Option 3: Using VS Code

1. Open the project folder in VS Code
2. Install the Jupyter extension
3. Open `Conversational_AI_Assignment_2_Gorup_45.ipynb`
4. Select Python kernel
5. Run all cells

### Execution Workflow

The notebook execution follows this sequence:

1. **Install Dependencies** (Cell 8)
2. **Import Libraries** (Cell 10)
3. **Configure Corpus Size** (Cell 12)
4. **Load Fixed URLs** (Cell 14)
5. **Fetch Wikipedia Articles** (Cell 34)
6. **Preprocess and Chunk Text** (Cell 36)
7. **Build Vector Index (FAISS)** (Cell 39)
8. **Build BM25 Index** (Cell 44)
9. **Generate Evaluation Questions** (Cell 47)
10. **Run Evaluation** (Cell 50-57)

### Key Configuration Parameters

In the notebook, you can adjust:

```python
FIXED_URL_COUNT = 200      # Fixed Wikipedia URLs
RANDOM_URL_COUNT = 300     # Random URLs per run
TOTAL_CORPUS_SIZE = 500    # Total corpus size
EVAL_QUESTION_COUNT = 100  # Questions for evaluation

# Retrieval parameters to testing purpose only.
TOP_K_DENSE = 10          # Top-K for dense retrieval
TOP_K_SPARSE = 10         # Top-K for sparse retrieval
TOP_N_RRF = 5             # Top-N after RRF fusion
RRF_K = 60                # RRF constant
```

## Evaluation Instructions

### Running the Automated Evaluation

The evaluation system automatically:
1. Generates 100 questions from the corpus
2. Retrieves relevant chunks using hybrid RAG
3. Generates answers using LLM
4. Calculates evaluation metrics

### Evaluation Metrics

The system calculates:

- **Mean Reciprocal Rank (MRR)**: Quality of retrieval ranking
- **Average F1 Score**: Answer quality vs ground truth
- **Recall@5**: Percentage of correct answers in top-5 chunks
- **Average Latency**: Response time per query
- **Total Pipeline Time**: Complete evaluation duration

### View Evaluation Results

Results are saved in:

1. **`evaluation_summary.json`**: Overall metrics
   ```json
   {
       "Mean Reciprocal Rank (MRR)": 0.6237,
       "Average F1 Score": 0.0906,
       "Recall@5 Rate": 0.7,
       "Avg Latency (s)": 3.70,
       "Total Pipeline Time (min)": 6.17
   }
   ```

2. **`evaluation_results_full.csv`**: Detailed per-question results
   - Question
   - Generated Answer
   - Ground Truth
   - Retrieved Chunks
   - Scores (Dense, Sparse, RRF)
   - Metrics (MRR, F1, Recall@5)
   - Latency

3. **Visualization**: The notebook generates plots showing:
   - MRR distribution
   - F1 Score distribution
   - Latency analysis
   - Recall@5 performance

## Project Files Structure

```
bitsMtech_CAI_Assignment-2/
├── Conversational_AI_Assignment_2_Gorup_45.ipynb  # Main notebook
├── README.md                                       # This file
├── fixed_urls.json                                # 200 fixed Wikipedia URLs
├── questions_100.json                             # Generated evaluation questions
├── preprocessed_corpus.json                       # Processed Wikipedia articles
├── vector_database.index                          # FAISS vector index
├── evaluation_results_full.csv                    # Detailed results
├── evaluation_summary.json                        # Summary metrics
├── ArchitecutureDiagram.drawio                    # System architecture
├── chroma-DB/                                      # Vector database files
│   └── chroma.sqlite3
└── Output SS/                                      # Output screenshots
```

## Fixed Wikipedia URLs (200 URLs)

Below is the complete list of 200 fixed Wikipedia URLs used in this project:

## Usage Example

### Query the System

Once the system is set up and running, you can query it:

```python
query = "What is machine learning?"
result = hybrid_rag_system.answer_query(query)

print("Answer:", result['answer'])
print("Top Sources:")
for chunk in result['retrieved_chunks']:
    print(f"- {chunk['source']} (Score: {chunk['rrf_score']:.4f})")
```

### Sample Output

```
Answer: Machine learning is a subset of artificial intelligence that enables 
computer systems to learn and improve from experience without being explicitly 
programmed. It focuses on the development of algorithms that can access data 
and use it to learn for themselves.

Top Sources:
- Machine Learning (Wikipedia) (Score: 0.9234)
- Artificial Intelligence (Wikipedia) (Score: 0.8567)
- Deep Learning (Wikipedia) (Score: 0.7892)
```

## Troubleshooting

### Common Issues

1. **NLTK Download Error**:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('punkt_tab')
   ```

2. **Memory Issues**: Reduce corpus size or chunk size in configuration

3. **FAISS Installation Issues**: 
   - Use `faiss-cpu` for CPU-only systems
   - Use `faiss-gpu` for GPU acceleration (requires CUDA)

4. **Wikipedia API Rate Limiting**: 
   - Add delays between requests
   - Use caching for repeated queries

## Performance Metrics

Based on our evaluation with 100 questions:

- **Mean Reciprocal Rank**: 0.624
- **Average F1 Score**: 0.091
- **Recall@5 Rate**: 70%
- **Average Latency**: 3.7 seconds
- **Total Pipeline Time**: ~6.2 minutes

## Future Improvements

- Implement caching for faster repeated queries
- Add support for multi-modal retrieval
- Optimize chunk size and overlap parameters
- Integrate more advanced LLMs (GPT-4, Claude, etc.)
- Add real-time query interface (Streamlit/Gradio)
- Implement query expansion and reranking

## References

- [FAISS Documentation]## License

This project is submitted as part of the Conversational AI course assignment at BITS Pilani.

## Contact

For questions or issues, please contact any of the team members listed above.
(https://faiss.ai/)
- [Sentence Transformers](https://www.sbert.net/)
- [BM25 Algorithm](https://en.wikipedia.org/wiki/Okapi_BM25)
- [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [Transformers Library](https://huggingface.co/docs/transformers)

---

**Date**: February 2026  
**Course**: Conversational AI  
**Institution**: BITS Pilani
