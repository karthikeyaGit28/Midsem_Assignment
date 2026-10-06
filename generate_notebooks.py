"""
Script to generate reproducible Jupyter notebooks:
1. notebooks/data_exploration.ipynb
2. notebooks/experiments.ipynb
"""

import json
import os

def create_notebook(cells, output_path):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.13"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Generated notebook {output_path}")

# 1. data_exploration.ipynb
cells_exp = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# LegalLens: Data Exploration & Schema Analysis\n",
            "This notebook explores the **IndicLegalQA** dataset (Indian Supreme Court Q&A benchmark).\n",
            "We inspect the raw schema, group records into canonical judgment documents, and analyze corpus statistics."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import json\n",
            "import os\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "from collections import Counter\n",
            "\n",
            "# Load raw dataset\n",
            "raw_path = '../data/raw/IndicLegalQA_Dataset_10K_Revised.json'\n",
            "if not os.path.exists(raw_path):\n",
            "    raw_path = '../IndicLegalQA Dataset_10K_Revised.json'\n",
            "\n",
            "with open(raw_path, 'r', encoding='utf-8') as f:\n",
            "    records = json.load(f)\n",
            "\n",
            "print(f'Total Q&A records: {len(records)}')\n",
            "print('Sample record keys:', list(records[0].keys()))"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Convert to DataFrame for descriptive analysis\n",
            "df = pd.DataFrame(records)\n",
            "print('Unique cases:', df['case_name'].nunique())\n",
            "print('Unique judgment dates:', df['judgement_date'].nunique())\n",
            "df['q_words'] = df['question'].apply(lambda x: len(str(x).split()))\n",
            "df['a_words'] = df['answer'].apply(lambda x: len(str(x).split()))\n",
            "df[['q_words', 'a_words']].describe()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load processed canonical documents\n",
            "with open('../data/processed/documents.json', 'r', encoding='utf-8') as f:\n",
            "    docs = json.load(f)\n",
            "\n",
            "df_docs = pd.DataFrame(docs)\n",
            "print(f'Total canonical documents: {len(df_docs)}')\n",
            "print('\\nLegal Category Distribution:')\n",
            "print(df_docs['category'].value_counts())\n",
            "\n",
            "# Plot category distribution\n",
            "plt.figure(figsize=(8, 4), dpi=150)\n",
            "df_docs['category'].value_counts().plot(kind='barh', color='#3f51b5')\n",
            "plt.title('Distribution of Legal Categories in LegalLens Corpus')\n",
            "plt.xlabel('Number of Judgments')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    }
]

# 2. experiments.ipynb
cells_expts = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# LegalLens: Information Retrieval Experiments & Evaluation\n",
            "This notebook runs and visualizes the benchmark comparing **TF-IDF**, **BM25**, **Dense Semantic**, and **Hybrid Retrieval**."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import sys\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "sys.path.insert(0, os.path.abspath('..'))\n",
            "from src.hybrid_ranker import load_hybrid_system\n",
            "from src.tfidf_retriever import TFIDFRetriever\n",
            "\n",
            "# Load models\n",
            "hybrid = load_hybrid_system()\n",
            "tfidf = TFIDFRetriever.load('../data/processed/tfidf_vectorizer.pkl')\n",
            "print('Models loaded successfully!')"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Test query on Hybrid Ranker\n",
            "query = 'Can a tenant be evicted without proper notice under rent control?'\n",
            "results = hybrid.rank_query(query, top_k=5, alpha=0.75)\n",
            "print(f'Query: {query}\\n')\n",
            "for i, r in enumerate(results, 1):\n",
            "    print(f\"{i}. {r['case_name']} (Final Score: {r['final_score']:.4f})\")\n",
            "    print(f\"   BM25 Norm: {r['bm25_norm']:.4f} | Semantic Norm: {r['semantic_norm']:.4f}\")\n",
            "    print(f\"   Passage: \\\"{r['passage'][:120]}...\\\"\\n\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Display Benchmark Results\n",
            "df_metrics = pd.read_csv('../results/metrics.csv')\n",
            "display(df_metrics)\n",
            "\n",
            "# Display Tuning Curve\n",
            "df_tuning = pd.read_csv('../results/alpha_tuning.csv')\n",
            "plt.figure(figsize=(8, 4), dpi=150)\n",
            "plt.plot(df_tuning['alpha'], df_tuning['nDCG@10'], marker='o', label='nDCG@10', color='#9013FE')\n",
            "plt.plot(df_tuning['alpha'], df_tuning['MRR'], marker='s', label='MRR', color='#4A90E2')\n",
            "plt.xlabel('Alpha Weight (BM25 vs Semantic)')\n",
            "plt.ylabel('Score')\n",
            "plt.title('Sensitivity Analysis of Hybrid Fusion Alpha')\n",
            "plt.legend()\n",
            "plt.grid(True)\n",
            "plt.show()"
        ]
    }
]

create_notebook(cells_exp, "notebooks/data_exploration.ipynb")
create_notebook(cells_expts, "notebooks/experiments.ipynb")
