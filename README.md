# Conversational AI Assignment 2 - Hybrid RAG System

**Group ID: 54**

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

# Retrieval parameters
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

```json
[
  "https://en.wikipedia.org/wiki/Artificial_intelligence",
  "https://en.wikipedia.org/wiki/Machine_learning",
  "https://en.wikipedia.org/wiki/Deep_learning",
  "https://en.wikipedia.org/wiki/Neural_network",
  "https://en.wikipedia.org/wiki/Natural_language_processing",
  "https://en.wikipedia.org/wiki/Computer_vision",
  "https://en.wikipedia.org/wiki/Reinforcement_learning",
  "https://en.wikipedia.org/wiki/Genetic_algorithm",
  "https://en.wikipedia.org/wiki/Expert_system",
  "https://en.wikipedia.org/wiki/Knowledge_representation",
  "https://en.wikipedia.org/wiki/Data_mining",
  "https://en.wikipedia.org/wiki/Big_data",
  "https://en.wikipedia.org/wiki/Pattern_recognition",
  "https://en.wikipedia.org/wiki/Information_retrieval",
  "https://en.wikipedia.org/wiki/Semantic_web",
  "https://en.wikipedia.org/wiki/Robotics",
  "https://en.wikipedia.org/wiki/Autonomous_robot",
  "https://en.wikipedia.org/wiki/Computer_science",
  "https://en.wikipedia.org/wiki/Algorithm",
  "https://en.wikipedia.org/wiki/Computational_complexity_theory",
  "https://en.wikipedia.org/wiki/Quantum_computing",
  "https://en.wikipedia.org/wiki/Operating_system",
  "https://en.wikipedia.org/wiki/Distributed_computing",
  "https://en.wikipedia.org/wiki/Cloud_computing",
  "https://en.wikipedia.org/wiki/Internet_of_things",
  "https://en.wikipedia.org/wiki/Cybersecurity",
  "https://en.wikipedia.org/wiki/Cryptography",
  "https://en.wikipedia.org/wiki/Blockchain",
  "https://en.wikipedia.org/wiki/Computer_network",
  "https://en.wikipedia.org/wiki/Database",
  "https://en.wikipedia.org/wiki/Software_engineering",
  "https://en.wikipedia.org/wiki/Programming_language",
  "https://en.wikipedia.org/wiki/Compiler",
  "https://en.wikipedia.org/wiki/Computer_architecture",
  "https://en.wikipedia.org/wiki/Embedded_system",
  "https://en.wikipedia.org/wiki/Human%E2%80%93computer_interaction",
  "https://en.wikipedia.org/wiki/Information_security",
  "https://en.wikipedia.org/wiki/Digital_signal_processing",
  "https://en.wikipedia.org/wiki/Image_processing",
  "https://en.wikipedia.org/wiki/Parallel_computing",
  "https://en.wikipedia.org/wiki/World_War_I",
  "https://en.wikipedia.org/wiki/World_War_II",
  "https://en.wikipedia.org/wiki/French_Revolution",
  "https://en.wikipedia.org/wiki/Industrial_Revolution",
  "https://en.wikipedia.org/wiki/Renaissance",
  "https://en.wikipedia.org/wiki/Cold_War",
  "https://en.wikipedia.org/wiki/Russian_Revolution",
  "https://en.wikipedia.org/wiki/American_Civil_War",
  "https://en.wikipedia.org/wiki/Ancient_Egypt",
  "https://en.wikipedia.org/wiki/Roman_Empire",
  "https://en.wikipedia.org/wiki/History_of_India",
  "https://en.wikipedia.org/wiki/Mughal_Empire",
  "https://en.wikipedia.org/wiki/British_Empire",
  "https://en.wikipedia.org/wiki/History_of_China",
  "https://en.wikipedia.org/wiki/History_of_Europe",
  "https://en.wikipedia.org/wiki/Feudalism",
  "https://en.wikipedia.org/wiki/Colonialism",
  "https://en.wikipedia.org/wiki/Imperialism",
  "https://en.wikipedia.org/wiki/Nationalism",
  "https://en.wikipedia.org/wiki/Global_conflict",
  "https://en.wikipedia.org/wiki/Indian_Constitution",
  "https://en.wikipedia.org/wiki/United_States_Constitution",
  "https://en.wikipedia.org/wiki/International_law",
  "https://en.wikipedia.org/wiki/Criminal_law",
  "https://en.wikipedia.org/wiki/Civil_law_(legal_system)",
  "https://en.wikipedia.org/wiki/Human_rights",
  "https://en.wikipedia.org/wiki/Intellectual_property",
  "https://en.wikipedia.org/wiki/Contract_law",
  "https://en.wikipedia.org/wiki/Tort_law",
  "https://en.wikipedia.org/wiki/Administrative_law",
  "https://en.wikipedia.org/wiki/Constitutional_law",
  "https://en.wikipedia.org/wiki/Legal_system",
  "https://en.wikipedia.org/wiki/Judiciary",
  "https://en.wikipedia.org/wiki/Rule_of_law",
  "https://en.wikipedia.org/wiki/Legal_history",
  "https://en.wikipedia.org/wiki/International_humanitarian_law",
  "https://en.wikipedia.org/wiki/Comparative_law",
  "https://en.wikipedia.org/wiki/Statutory_law",
  "https://en.wikipedia.org/wiki/Common_law",
  "https://en.wikipedia.org/wiki/Civil_procedure",
  "https://en.wikipedia.org/wiki/Economics",
  "https://en.wikipedia.org/wiki/Microeconomics",
  "https://en.wikipedia.org/wiki/Macroeconomics",
  "https://en.wikipedia.org/wiki/Inflation",
  "https://en.wikipedia.org/wiki/Unemployment",
  "https://en.wikipedia.org/wiki/Monetary_policy",
  "https://en.wikipedia.org/wiki/Fiscal_policy",
  "https://en.wikipedia.org/wiki/International_trade",
  "https://en.wikipedia.org/wiki/Globalization",
  "https://en.wikipedia.org/wiki/Behavioral_economics",
  "https://en.wikipedia.org/wiki/Development_economics",
  "https://en.wikipedia.org/wiki/Public_finance",
  "https://en.wikipedia.org/wiki/Economic_growth",
  "https://en.wikipedia.org/wiki/Market_economy",
  "https://en.wikipedia.org/wiki/Capitalism",
  "https://en.wikipedia.org/wiki/Socialism",
  "https://en.wikipedia.org/wiki/Mixed_economy",
  "https://en.wikipedia.org/wiki/Game_theory",
  "https://en.wikipedia.org/wiki/Labor_economics",
  "https://en.wikipedia.org/wiki/Financial_market",
  "https://en.wikipedia.org/wiki/Physics",
  "https://en.wikipedia.org/wiki/Classical_mechanics",
  "https://en.wikipedia.org/wiki/Quantum_mechanics",
  "https://en.wikipedia.org/wiki/Thermodynamics",
  "https://en.wikipedia.org/wiki/Electromagnetism",
  "https://en.wikipedia.org/wiki/Relativity",
  "https://en.wikipedia.org/wiki/Astrophysics",
  "https://en.wikipedia.org/wiki/Particle_physics",
  "https://en.wikipedia.org/wiki/Nuclear_physics",
  "https://en.wikipedia.org/wiki/Optics",
  "https://en.wikipedia.org/wiki/Chemistry",
  "https://en.wikipedia.org/wiki/Organic_chemistry",
  "https://en.wikipedia.org/wiki/Inorganic_chemistry",
  "https://en.wikipedia.org/wiki/Physical_chemistry",
  "https://en.wikipedia.org/wiki/Biochemistry",
  "https://en.wikipedia.org/wiki/Periodic_table",
  "https://en.wikipedia.org/wiki/Chemical_bond",
  "https://en.wikipedia.org/wiki/Catalysis",
  "https://en.wikipedia.org/wiki/Polymer",
  "https://en.wikipedia.org/wiki/Reaction_rate",
  "https://en.wikipedia.org/wiki/Biology",
  "https://en.wikipedia.org/wiki/Genetics",
  "https://en.wikipedia.org/wiki/Evolution",
  "https://en.wikipedia.org/wiki/Cell_biology",
  "https://en.wikipedia.org/wiki/Molecular_biology",
  "https://en.wikipedia.org/wiki/Ecology",
  "https://en.wikipedia.org/wiki/Neuroscience",
  "https://en.wikipedia.org/wiki/Immunology",
  "https://en.wikipedia.org/wiki/Microbiology",
  "https://en.wikipedia.org/wiki/Biotechnology",
  "https://en.wikipedia.org/wiki/Medicine",
  "https://en.wikipedia.org/wiki/Public_health",
  "https://en.wikipedia.org/wiki/Epidemiology",
  "https://en.wikipedia.org/wiki/Cardiology",
  "https://en.wikipedia.org/wiki/Neurology",
  "https://en.wikipedia.org/wiki/Oncology",
  "https://en.wikipedia.org/wiki/Pediatrics",
  "https://en.wikipedia.org/wiki/Psychiatry",
  "https://en.wikipedia.org/wiki/Pharmacology",
  "https://en.wikipedia.org/wiki/Surgery",
  "https://en.wikipedia.org/wiki/Climate_change",
  "https://en.wikipedia.org/wiki/Global_warming",
  "https://en.wikipedia.org/wiki/Renewable_energy",
  "https://en.wikipedia.org/wiki/Solar_energy",
  "https://en.wikipedia.org/wiki/Wind_power",
  "https://en.wikipedia.org/wiki/Hydroelectricity",
  "https://en.wikipedia.org/wiki/Nuclear_power",
  "https://en.wikipedia.org/wiki/Sustainable_development",
  "https://en.wikipedia.org/wiki/Environmental_science",
  "https://en.wikipedia.org/wiki/Biodiversity",
  "https://en.wikipedia.org/wiki/Geography",
  "https://en.wikipedia.org/wiki/Physical_geography",
  "https://en.wikipedia.org/wiki/Human_geography",
  "https://en.wikipedia.org/wiki/Geology",
  "https://en.wikipedia.org/wiki/Plate_tectonics",
  "https://en.wikipedia.org/wiki/Volcano",
  "https://en.wikipedia.org/wiki/Earthquake",
  "https://en.wikipedia.org/wiki/Meteorology",
  "https://en.wikipedia.org/wiki/Oceanography",
  "https://en.wikipedia.org/wiki/Cartography",
  "https://en.wikipedia.org/wiki/Federated_learning",
  "https://en.wikipedia.org/wiki/Self-supervised_learning",
  "https://en.wikipedia.org/wiki/Zero-shot_learning",
  "https://en.wikipedia.org/wiki/Few-shot_learning",
  "https://en.wikipedia.org/wiki/Knowledge_graph",
  "https://en.wikipedia.org/wiki/Search_engine",
  "https://en.wikipedia.org/wiki/Information_extraction",
  "https://en.wikipedia.org/wiki/Question_answering",
  "https://en.wikipedia.org/wiki/Text_mining",
  "https://en.wikipedia.org/wiki/Sentiment_analysis",
  "https://en.wikipedia.org/wiki/History_of_computing",
  "https://en.wikipedia.org/wiki/Computer_program",
  "https://en.wikipedia.org/wiki/Debugging",
  "https://en.wikipedia.org/wiki/Software_testing",
  "https://en.wikipedia.org/wiki/Agile_software_development",
  "https://en.wikipedia.org/wiki/Philosophy_of_artificial_intelligence",
  "https://en.wikipedia.org/wiki/Artificial_general_intelligence",
  "https://en.wikipedia.org/wiki/Machine_ethics",
  "https://en.wikipedia.org/wiki/Automation",
  "https://en.wikipedia.org/wiki/Control_system",
  "https://en.wikipedia.org/wiki/Political_economy",
  "https://en.wikipedia.org/wiki/Welfare_state",
  "https://en.wikipedia.org/wiki/Public_policy",
  "https://en.wikipedia.org/wiki/Regulation",
  "https://en.wikipedia.org/wiki/Corporate_governance",
  "https://en.wikipedia.org/wiki/International_organization",
  "https://en.wikipedia.org/wiki/Non-governmental_organization",
  "https://en.wikipedia.org/wiki/United_Nations_Development_Programme",
  "https://en.wikipedia.org/wiki/World_Bank",
  "https://en.wikipedia.org/wiki/International_Labour_Organization",
  "https://en.wikipedia.org/wiki/Energy_economics",
  "https://en.wikipedia.org/wiki/Climate_policy",
  "https://en.wikipedia.org/wiki/Carbon_tax",
  "https://en.wikipedia.org/wiki/Emission_trading",
  "https://en.wikipedia.org/wiki/Circular_economy",
  "https://en.wikipedia.org/wiki/History_of_technology",
  "https://en.wikipedia.org/wiki/Industrial_design",
  "https://en.wikipedia.org/wiki/Innovation",
  "https://en.wikipedia.org/wiki/Technological_change",
  "https://en.wikipedia.org/wiki/Science_and_technology_studies"
]
```

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

- [FAISS Documentation](https://faiss.ai/)
- [Sentence Transformers](https://www.sbert.net/)
- [BM25 Algorithm](https://en.wikipedia.org/wiki/Okapi_BM25)
- [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [Transformers Library](https://huggingface.co/docs/transformers)

## License

This project is submitted as part of the Conversational AI course assignment at BITS Pilani.

## Contact

For questions or issues, please contact any of the team members listed above.

---

**Date**: February 2026  
**Course**: Conversational AI  
**Institution**: BITS Pilani
