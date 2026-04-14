# Multimodal Short-Video Recommendation System

This project builds a recommendation system to improve content relevance on a mobile short-video platform. The system predicts user engagement with candidate videos, measured by **watch ratio**, and uses it to rank and recommend the most relevant content to users.

The project explores and combines multiple recommendation paradigms — **content-based filtering**, **collaborative filtering (NeuMF)**, **graph-based (LightGCN)**, and **sequential (SASRec)** models — culminating in a **two-stage candidate generation and re-ranking pipeline**.

## Dataset

The project uses the [ShortVideo dataset](https://github.com/tsinghua-fib-lab/ShortVideo_dataset) from Tsinghua FIB Lab, which contains interaction logs, video categories, titles, ASR transcriptions, and visual features from a real-world short-video platform.

**All datasets (raw and processed splits) can be found here:**
[Google Drive — bt4222-group9-data](https://drive.google.com/drive/folders/1KiImqYViQ__yzpglCbTJICX1HcBHMUxw)

The Drive folder is organized as follows:

| Folder | Contents |
|--------|----------|
| `raw_datasets/` | Original data files from the ShortVideo dataset (`interaction.csv`, `categories_cn_en.csv`, per-video title/ASR/feature files) |
| `processed_split/` | Train/validation/test splits (engineered and scaled CSVs), enriched entity tables |
| `artifacts/` | PCA models, scalers, label encoders, embedding matrices, model checkpoints, and metadata |

---

## Pipeline Overview

The project is structured as a sequential notebook pipeline. Each notebook reads artifacts produced by earlier steps and writes outputs consumed by later steps.

1. Data Loading
2. Exploratory Data Analysis (EDA)
3. Data Preprocessing
4. Caption-based Filtering
5. NeuMF Model Training, Hybrid Model & Evaluation
6. SASRec Two Stage Filtering & Evaluation

---

## Source Code — Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `1_data_loading.ipynb` | Ingests raw interaction logs, categories, per-video titles, ASR text, and video embeddings from the ShortVideo dataset; merges them into unified analysis-ready tables |
| 2 | `2_eda.ipynb` | Exploratory data analysis on users, videos, interactions, and the watch ratio target variable |
| 3 | `3_data_preprocessing.ipynb` | Feature engineering, train/val/test splitting, BERT text encoding with PCA reduction, label encoding, and standard scaling; produces all data artifacts used downstream |
| 4 | `4_caption_based_filtering.ipynb` | Content-based filtering baseline — builds user profiles from watch-ratio-weighted and time-decayed text embeddings, ranks videos by cosine similarity |
| 5a | `5_nmf_scaled.ipynb` | Neural Matrix Factorization (NeuMF) — ID-only and feature-enhanced variants trained on **scaled** features, with Ridge-based hybrid evaluation |
| 5b | `5_nmf_unscaled.ipynb` | Same NeuMF pipeline trained on **engineered (unscaled)** features for comparison |
| 6 | `6_sasrec_twostage.ipynb` | Sequential recommendation with SASRec, LightGCN baseline, and a two-stage LightGCN → SASRec re-ranking pipeline; includes BPR-SASRec with side features |

---

## Data Files — Complete Inventory

### Raw Data (External / `raw_datasets/`)

These files come from the [ShortVideo dataset](https://github.com/tsinghua-fib-lab/ShortVideo_dataset) and are the starting point of the pipeline.

| File | Description | Used by |
|------|-------------|---------|
| `interaction.csv` | User-video interaction logs (timestamps, watch time, duration) | Notebook 1, 3 |
| `categories_cn_en.csv` | Video category mappings (Chinese ↔ English) | Notebook 1, 2, 3 |
| `title_en/{pid}.txt` | Per-video English titles (downloaded in Notebook 1) | Notebook 1 |
| `asr_en/{pid}.txt` | Per-video English ASR transcriptions (downloaded in Notebook 1) | Notebook 1 |
| `video_feature_total/{pid}.npy` | Per-video visual feature embeddings (downloaded in Notebook 1) | Notebook 1 |

### Processed Entity Tables (produced by Notebook 1 → used by Notebooks 2, 3, 6)

| File | Description | Produced by | Used by |
|------|-------------|-------------|---------|
| `behavior_data.csv` | Cleaned interaction records with user-video pairs | Notebook 1 | Notebook 2, 3 |
| `user_data.csv` | User-level attributes | Notebook 1 | Notebook 2, 3, 6 |
| `video_data.csv` | Video-level attributes (categories, duration) | Notebook 1 | Notebook 2, 3, 6 |
| `video_data_enriched.csv` | Video data with appended title and ASR text columns | Notebook 1 | — |
| `video_data_enriched.pkl` | Pickle version of the enriched video table | Notebook 1 | — |
| `video_data_full.csv` | Final merged video table with all text and visual features | Notebook 1 | Notebook 3 |

### Train/Val/Test Splits (produced by Notebook 3 → used by Notebooks 4, 5, 6)

| File | Description | Used by |
|------|-------------|---------|
| `train_engineered.csv` | Training set with engineered features (unscaled) | Notebook 4, 5b, 6 |
| `val_engineered.csv` | Validation set with engineered features (unscaled) | Notebook 4, 5b, 6 |
| `test_engineered.csv` | Test set with engineered features (unscaled) | Notebook 4, 5b, 6 |
| `train_scaled.csv` | Training set with standardized features | Notebook 5a |
| `val_scaled.csv` | Validation set with standardized features | Notebook 5a |
| `test_scaled.csv` | Test set with standardized features | Notebook 5a |
| `video_data_engineered.csv` | Video entity table with engineered features | Notebook 6 (fallback) |
| `user_data_engineered.csv` | User entity table with engineered features | — |

### Artifacts — Encoders, Scalers & Metadata (produced by Notebook 3)

| File | Description | Used by |
|------|-------------|---------|
| `artifacts/model_meta.pkl` | Central metadata dict (number of users/videos, PCA dimensions, feature counts) | Notebook 4, 5a, 5b |
| `artifacts/pid_label_map.pkl` | Video ID → label index mapping | Notebook 3 (internal) |
| `artifacts/user_label_map.pkl` | User ID → label index mapping | Notebook 3 (internal) |
| `artifacts/le_city.pkl` | LabelEncoder for city feature | — |
| `artifacts/le_root.pkl` | LabelEncoder for root category feature | — |
| `artifacts/interaction_scaler.pkl` | StandardScaler fitted on interaction features | — |
| `artifacts/user_scaler.pkl` | StandardScaler fitted on user features | — |
| `artifacts/video_scaler.pkl` | StandardScaler fitted on video features | — |

### Artifacts — Text Embeddings (produced by Notebook 3)

768-dimensional BERT embeddings (one row per video):

| File | Description | Used by |
|------|-------------|---------|
| `artifacts/emb_title_cn_768.npy` | Chinese title embeddings | Notebook 3 (PCA input) |
| `artifacts/emb_tags_cn_768.npy` | Chinese tag embeddings | Notebook 3 (PCA input) |
| `artifacts/emb_title_en_768.npy` | English title embeddings | Notebook 3 (PCA input) |
| `artifacts/emb_asr_en_768.npy` | English ASR embeddings | Notebook 3 (PCA input) |

PCA-reduced embeddings and fitted PCA models:

| File | Description | Used by |
|------|-------------|---------|
| `artifacts/pca_title_cn_128.npy` | Chinese title embeddings reduced to 128-d | Notebook 4, 5a, 5b |
| `artifacts/pca_title_cn_128.pkl` | Fitted PCA model for Chinese titles | — |
| `artifacts/pca_tags_cn_64.npy` | Chinese tag embeddings reduced to 64-d | Notebook 4, 5a, 5b |
| `artifacts/pca_tags_cn_64.pkl` | Fitted PCA model for Chinese tags | — |
| `artifacts/pca_title_en_128.npy` | English title embeddings reduced to 128-d | Notebook 4 |
| `artifacts/pca_title_en_128.pkl` | Fitted PCA model for English titles | — |
| `artifacts/pca_asr_en_128.npy` | English ASR embeddings reduced to 128-d | Notebook 4 |
| `artifacts/pca_asr_en_128.pkl` | Fitted PCA model for English ASR | — |

### Artifacts — Feature Matrices (produced by Notebook 3)

| File | Description | Used by |
|------|-------------|---------|
| `artifacts/user_continuous_matrix.npy` | Dense matrix of continuous user features | Notebook 5a, 5b |
| `artifacts/video_continuous_matrix.npy` | Dense matrix of continuous video features | Notebook 5a, 5b |
| `artifacts/user_fre_city_encoded.npy` | Encoded categorical user features (frequency, city) | Notebook 5a, 5b |
| `artifacts/video_root_id_encoded.npy` | Encoded categorical video features (root category) | Notebook 5a, 5b |

### Artifacts — Model Checkpoints (produced by Notebooks 5 & 6)

| File | Description | Produced by |
|------|-------------|-------------|
| `artifacts/model_a_best.pt` | Best NeuMF Model A (ID-only) weights | Notebook 5a / 5b |
| `artifacts/model_b_best.pt` | Best NeuMF Model B (feature-enhanced) weights | Notebook 5a / 5b |
| `artifacts/model_c_best.pt` | Best NeuMF Model C (feature-enhanced variant) weights | Notebook 5a / 5b |
| `artifacts/model_a_cfg.pkl` | Hyperparameters for Model A | Notebook 5a |
| `artifacts/model_b_cfg.pkl` | Hyperparameters for Model B | Notebook 5a |
| `artifacts/model_c_cfg.pkl` | Hyperparameters for Model C | Notebook 5a |
| `artifacts/best_neumf_key.pkl` | Key identifying which NeuMF variant performed best | Notebook 5a |
| `artifacts/hybrid_ridge.pkl` | Fitted Ridge regression model for hybrid scoring | Notebook 5a / 5b |
| `artifacts/cbf_config.pkl` | Best content-based filtering configuration | Notebook 5a |
| `artifacts/pca_cn_scaler.pkl` | Scaler for PCA Chinese embeddings | Notebook 5a / 5b |
| `artifacts/pca_tags_scaler.pkl` | Scaler for PCA tag embeddings | Notebook 5a / 5b |
| `artifacts2/sasrec_idonly_best.pt` | Best SASRec (ID-only) weights | Notebook 6 |
| `artifacts2/bpr_sasrec_best.pt` | Best BPR-SASRec (with side features) weights | Notebook 6 |
| `artifacts2/lightgcn_candidates_top200.csv` | Top-200 LightGCN candidate lists per user (for two-stage re-ranking) | Notebook 6 |

### Output Plots

| File | Description | Produced by |
|------|-------------|-------------|
| `caption_stage1_ablation.png` | Ablation study over text modalities | Notebook 4 |
| `caption_all_experiments.png` | All caption-based filtering experiments | Notebook 4 |
| `caption_lambda_tuning.png` | Time-decay lambda hyperparameter tuning | Notebook 4 |
| `caption_filtering_summary.png` | Summary of caption-based filtering results | Notebook 4 |
| `neumf_comparison.png` | NeuMF model variant comparison | Notebook 5a / 5b |
| `final_model_comparison.png` | Final model comparison across all approaches | Notebook 5a / 5b |
| `artifacts2/sasrec_training_curves.png` | SASRec training loss/metric curves | Notebook 6 |
| `artifacts2/comparison_all_models.png` | Comparison of all sequential/two-stage models | Notebook 6 |

---

## Data Flow Between Notebooks

The diagram below shows how files flow from one notebook to the next. Each arrow represents a file dependency.

```
EXTERNAL DATASET (ShortVideo)
│
│  interaction.csv, categories_cn_en.csv
│  title_en/*.txt, asr_en/*.txt, video_feature_total/*.npy
│
▼
┌─────────────────────────────────────────────────────────┐
│  NOTEBOOK 1 — Data Loading                              │
│                                                         │
│  Reads: interaction.csv, categories_cn_en.csv,          │
│         title_en/, asr_en/, video_feature_total/        │
│                                                         │
│  Writes: behavior_data.csv                              │
│          user_data.csv                                  │
│          video_data.csv                                 │
│          video_data_full.csv                            │
│          video_data_enriched.csv / .pkl                 │
└──────────────┬──────────────────────────────────────────┘
               │
               │  behavior_data.csv, user_data.csv,
               │  video_data.csv, categories_cn_en.csv
               ▼
┌─────────────────────────────────────────────────────────┐
│  NOTEBOOK 2 — EDA                                       │
│                                                         │
│  Reads: behavior_data.csv, user_data.csv,               │
│         video_data.csv, categories_cn_en.csv            │
│                                                         │
│  Writes: (plots only, no data artifacts)                │
└─────────────────────────────────────────────────────────┘

               │  behavior_data.csv, user_data.csv,
               │  video_data.csv, video_data_full.csv,
               │  interaction.csv, categories_cn_en.csv
               ▼
┌─────────────────────────────────────────────────────────┐
│  NOTEBOOK 3 — Data Preprocessing                        │
│                                                         │
│  Reads: behavior_data.csv, user_data.csv,               │
│         video_data.csv, video_data_full.csv,            │
│         interaction.csv, categories_cn_en.csv           │
│                                                         │
│  Writes: train/val/test_engineered.csv                  │
│          train/val/test_scaled.csv                      │
│          user_data_engineered.csv                       │
│          video_data_engineered.csv                      │
│          artifacts/model_meta.pkl                       │
│          artifacts/emb_*_768.npy  (BERT embeddings)     │
│          artifacts/pca_*_*.npy    (PCA embeddings)      │
│          artifacts/pca_*_*.pkl    (PCA models)          │
│          artifacts/user_continuous_matrix.npy            │
│          artifacts/video_continuous_matrix.npy           │
│          artifacts/user_fre_city_encoded.npy             │
│          artifacts/video_root_id_encoded.npy             │
│          artifacts/*_scaler.pkl, artifacts/le_*.pkl      │
│          artifacts/*_label_map.pkl                       │
└──────────────┬──────────────────────────────────────────┘
               │
       ┌───────┼────────────────────┐
       │       │                    │
       ▼       ▼                    ▼
┌───────────┐ ┌──────────────┐ ┌────────────────────┐
│ NB 4      │ │ NB 5a / 5b   │ │ NB 6               │
│ Caption   │ │ NeuMF        │ │ SASRec & Two-Stage  │
│ Filtering │ │              │ │                     │
│           │ │              │ │                     │
│ Reads:    │ │ Reads:       │ │ Reads:              │
│ model_    │ │ model_       │ │ train/val/test_     │
│ meta.pkl  │ │ meta.pkl     │ │ engineered.csv      │
│ pca_*.npy │ │ train/val/   │ │ user_data.csv       │
│ train/val/│ │ test_scaled  │ │ video_data.csv      │
│ test_eng- │ │ or _eng-     │ │ lightgcn_candidates │
│ ineered   │ │ ineered.csv  │ │ _top200.csv         │
│ .csv      │ │ *_matrix.npy │ │                     │
│           │ │ *_encoded.npy│ │ Writes:             │
│ Writes:   │ │ pca_*.npy    │ │ sasrec_idonly_      │
│ plots     │ │              │ │ best.pt             │
│ (.png)    │ │ Writes:      │ │ bpr_sasrec_best.pt  │
│           │ │ model_*_     │ │ lightgcn_candidates │
│           │ │ best.pt      │ │ _top200.csv         │
│           │ │ model_*_     │ │ plots (.png)        │
│           │ │ cfg.pkl      │ │                     │
│           │ │ hybrid_      │ │                     │
│           │ │ ridge.pkl    │ │                     │
│           │ │ plots (.png) │ │                     │
└───────────┘ └──────────────┘ └────────────────────┘
```

---

## Models

| Model | Type | Description |
|-------|------|-------------|
| **Content-Based Filtering** | Text similarity | Builds user profiles from watch-ratio-weighted and time-decayed BERT embeddings; ranks videos by cosine similarity |
| **NeuMF** | Collaborative filtering | Neural Matrix Factorization combining GMF and MLP pathways; extended with user/video side features |
| **LightGCN** | Graph-based | Lightweight graph convolution on the user-item interaction graph for candidate generation |
| **SASRec** | Sequential | Self-Attentive Sequential Recommendation using transformer blocks over time-ordered interaction sequences |
| **Two-Stage Pipeline** | Hybrid | LightGCN generates top-200 candidates; SASRec re-ranks them for final recommendation |

## Evaluation Metrics

- **NDCG@K** (Normalized Discounted Cumulative Gain)
- **Recall@K**
- **Precision@K**
- **MSE / RMSE** (for watch ratio prediction)

## Tech Stack

- **Python** (Jupyter Notebooks)
- **PyTorch** — deep learning models (NeuMF, SASRec, LightGCN)
- **Hugging Face Transformers** — BERT-based text encoding
- **scikit-learn** — PCA, StandardScaler, LabelEncoder, Ridge regression
- **pandas / NumPy** — data manipulation
- **matplotlib / seaborn** — visualization
- **Google Colab** — primary execution environment (notebooks use Drive-mounted paths)

## Getting Started

### Prerequisites

- Python 3.8+
- A Google Colab environment (recommended) or a local setup with GPU support

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/bt4222-project.git
   ```

2. Download the data from [Google Drive](https://drive.google.com/drive/folders/1KiImqYViQ__yzpglCbTJICX1HcBHMUxw) and place it in your Google Drive under the expected paths, or adjust the data directory paths in each notebook.

3. Install the required packages (within each notebook or manually):
   ```bash
   pip install torch transformers scikit-learn pandas numpy matplotlib seaborn tqdm joblib emoji beautifulsoup4 requests
   ```

4. Run the notebooks in order (1 → 2 → 3 → 4/5/6) to reproduce the full pipeline. Notebooks 4, 5, and 6 can be run independently of each other after Notebook 3 is complete.

## Project Structure

```
bt4222-project/
├── 1_data_loading.ipynb            # Data ingestion and multimodal assembly
├── 2_eda.ipynb                     # Exploratory data analysis
├── 3_data_preprocessing.ipynb      # Feature engineering and splitting
├── 4_caption_based_filtering.ipynb # Content-based filtering baseline
├── 5_nmf_scaled.ipynb              # NeuMF on scaled features
├── 5_nmf_unscaled.ipynb            # NeuMF on unscaled features
├── 6_sasrec_twostage.ipynb         # SASRec and two-stage pipeline
└── README.md
```
