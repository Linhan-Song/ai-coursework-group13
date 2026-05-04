# ai-coursework-group13

Coursework for **EMATM0067 — Introduction to AI and Text Analytics** (University of Bristol).  
This repository contains materials for **Task 3: comparative corpus analysis** of ML-related arXiv abstracts over time.

**Active development branch:** [`bryan_branch`](https://github.com/Linhan-Song/ai-coursework-group13/tree/bryan_branch)

---

## Repository layout

| File | Description |
|------|-------------|
| **`Bryan's_AI_task3_updated.ipynb`** | Main Jupyter notebook: preprocessing, TF–IDF / LDA / BERTopic (plus supplementary HDP checks), plots, and metric summaries. |
| **`Task3_report_draft.md`** | Draft report text aligned with the notebook outputs. |
| **`README.md`** | This file. |

---

## Focus of work on `bryan_branch` (Bo-Yan Lu)

Most of the later work emphasises **hyperparameter tuning** and **selecting stronger settings using quantitative criteria**, so comparisons are supported by repeatable experiments rather than one-off runs.

### LDA

- **Stopword regimes (baseline vs stricter filtering):** compare **coherence** and **distinctness** to see whether stricter stopword removal consistently improves topic quality across time bins.  
- **`K` (number of topics) and `passes` grid search:** search over candidate configurations and pick **period-specific** best settings by coherence (the best `K` is **not** assumed to be identical across bins because corpus size and topical diversity differ).

### BERTopic

- Compare multiple configuration variants (e.g. **HDBSCAN baseline**, **outlier reduction**, **KMeans**, **alternative sentence embeddings**).  
- Report **outlier rate (topic −1)** together with qualitative topic readability to explain the **coverage vs specificity** trade-off and which setup is most suitable for stable cross-period storytelling.

### TF–IDF

- Compare vectoriser settings such as **`max_features`**, **`min_df`**, and **`max_df`** to keep term statistics stable while avoiding vocabulary choices that either erase informative terms or dilute signal.

### Cross-method principle

- Keep the **evaluation frame fixed** (same temporal split; preprocessing matched to each model family), then tune **within** each method.  
- Use **metrics + qualitative inspection** (are top terms / topic keywords interpretable and consistent with plausible ML history?) when choosing what to present as the “best” configuration.

---

## How to run (brief)

1. Use a Python environment compatible with the notebook imports (see install cells at the top of `Bryan's_AI_task3_updated.ipynb`).  
2. Open **`Bryan's_AI_task3_updated.ipynb`** in Jupyter / VS Code / Cursor and run top-to-bottom.  
3. Hugging Face dataset downloads require **internet** on first run.

---


