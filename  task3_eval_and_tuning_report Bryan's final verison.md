# Task 3 — Topic Modelling and Temporal Analysis of ML arXiv Abstracts

---

## 1. Abstract

This report analyses machine-learning-related arXiv abstracts across three time periods: **1999–2010**, **2011–2015**, and **2016–2021**. We apply three main methods — **TF–IDF**, **LDA**, and **BERTopic** — to understand how research topics have shifted over time. An additional method, **HDP**, is included briefly as a supplementary check.

Key findings:

- **TF–IDF** reveals that terms like *neural*, *deep*, *graph*, and *network* become increasingly prominent in later periods.
- **LDA with strict stopwords** consistently outperforms the baseline stopword list, achieving higher coherence and distinctness in every period. The best single configuration is **(K=10, passes=30)** in the 2016–2021 period, with coherence **0.5215** and distinctness **0.9775**.
- **BERTopic** highlights a trade-off between coverage and specificity: the Baseline variant leaves ~31% of documents unassigned (outlier topic -1), while KMeans and Reduce-outliers assign all documents to topics, and MPNet sits at ~17% outliers.

Note: Topic IDs are **not aligned** across methods — we compare methods by examining **term-level trends**, **metric scores**, and **qualitative theme shifts**, not by matching topic numbers.

---

## 2. Data and Preprocessing

- **Source**: Hugging Face arXiv abstracts dataset (`gfissore/arxiv-abstracts-2021`), totalling approximately **7,015 documents**.
- **Time periods**: Documents are grouped into three bins based on publication year: **1999–2010**, **2011–2015**, and **2016–2021**. The latest period contains the majority of documents, so results for the earliest bin should be interpreted with caution due to its smaller size.
- **Text cleaning**:
  - For TF–IDF and LDA: a shared preprocessing function (`preprocess_for_tfidf_lda`) tokenises and cleans text.
  - For BERTopic: a separate cleaning function (`clean_text_embed`) prepares text for transformer-based embeddings.
- **Stopword settings for LDA**:
  - **Baseline**: standard sklearn English stopwords.
  - **Strict**: baseline + additional domain-generic terms (e.g., *method*, *approach*, *result*) removed to help topics focus on more meaningful, discriminative vocabulary.

---

## 3. Methods

### 3.1 TF–IDF (scikit-learn)

TF–IDF (Term Frequency–Inverse Document Frequency) converts each document into a numerical vector where each value reflects how important a word is to that document relative to the entire corpus. Common words get lower scores; distinctive words get higher scores.

- We use `TfidfVectorizer` with parameters `max_features`, `min_df`, and `max_df` to control vocabulary size.
- TF–IDF does not produce "topics" directly, but shows which terms characterise each time period by examining the highest-weighted terms.

### 3.2 LDA (Gensim)

LDA (Latent Dirichlet Allocation) is a generative topic model. It assumes each document is a mixture of topics, and each topic is a distribution over words. The model learns these distributions from the data.

**How it works in our pipeline:**

1. **Build a dictionary**: scan all tokens in a period and assign each unique word an ID. Words that are too rare or too common are filtered out (`filter_extremes`).
2. **Convert to bag-of-words**: each document becomes a list of (word ID, count) pairs using `doc2bow`. This is the input format LDA expects.
3. **Train LDA**: one separate model is trained per time period (not one global model), so topic indices **cannot be compared** across periods — only the keywords and themes can be compared qualitatively.

**Parameters varied:**

- `num_topics` (K): how many topics to discover.
- `passes`: how many times the algorithm iterates over the corpus during training.

**Evaluation metrics:**

- **Coherence**: measures how semantically related the top words in a topic are. Higher is better.
- **Distinctness**: measures how different topics are from each other (calculated as the mean pairwise 1 − Jaccard similarity of top-word sets). Higher means less overlap between topics.

### 3.3 BERTopic

BERTopic uses transformer-based sentence embeddings (e.g., MiniLM, MPNet) to represent documents as dense vectors, then clusters them to discover topics.

We test four variants:


| Variant             | Description                                                                          |
| ------------------- | ------------------------------------------------------------------------------------ |
| **Baseline**        | MiniLM embeddings + HDBSCAN clustering                                               |
| **Reduce outliers** | Same as Baseline, then reassigns outlier documents to the nearest topic              |
| **KMeans**          | Replaces HDBSCAN with KMeans clustering (fixed number of clusters)                   |
| **MPNet**           | Uses MPNet embeddings (a different, larger transformer) with the same HDBSCAN recipe |


Each variant is fitted independently, so topic IDs are only meaningful within that specific run.

### 3.4 HDP (supplementary)

HDP (Hierarchical Dirichlet Process) is a non-parametric variant of LDA that automatically infers the number of topics. It is included only as a sanity check — many of its inferred topics carry very little weight and are not individually meaningful.

---

## 4. Results

### 4.1 TF–IDF Parameter Sweep

We tested three vectoriser configurations (all using the same stopword list, varying only vocabulary-related parameters):

**Table 4.1a — Vectoriser settings and matrix shape**


| Config          | `max_features` | `max_df` | `min_df` | Vocab size | Non-zero entries | Density |
| --------------- | -------------- | -------- | -------- | ---------- | ---------------- | ------- |
| **default**     | 5,000          | 0.50     | 5        | 5,000      | 426,815          | 0.0122  |
| **more_vocab**  | 8,000          | 0.50     | 3        | 7,778      | 438,589          | 0.0080  |
| **stricter_df** | 5,000          | 0.35     | 8        | 4,642      | 416,960          | 0.0128  |


- *Non-zero entries* = number of cells in the document–term matrix that are not zero.
- *Density* = non-zero entries / (number of documents × vocabulary size). Lower density means a sparser matrix.

**Table 4.1b — Mean TF–IDF scores for selected terms, by config and period**


| Config          | Period    | neural | deep   | graph  | training | network |
| --------------- | --------- | ------ | ------ | ------ | -------- | ------- |
| **default**     | 1999–2010 | 0.0029 | 0.0000 | 0.0052 | —        | —       |
| **default**     | 2011–2015 | 0.0067 | 0.0097 | 0.0105 | 0.0146   | 0.0132  |
| **default**     | 2016–2021 | 0.0238 | 0.0204 | 0.0263 | 0.0236   | 0.0341  |
| **more_vocab**  | 1999–2010 | 0.0026 | 0.0000 | 0.0052 | —        | —       |
| **more_vocab**  | 2011–2015 | 0.0065 | 0.0093 | 0.0103 | 0.0142   | 0.0128  |
| **more_vocab**  | 2016–2021 | 0.0231 | 0.0197 | 0.0256 | 0.0228   | 0.0330  |
| **stricter_df** | 1999–2010 | 0.0029 | 0.0000 | 0.0052 | —        | —       |
| **stricter_df** | 2011–2015 | 0.0068 | 0.0098 | 0.0108 | 0.0149   | *NaN*   |
| **stricter_df** | 2016–2021 | 0.0244 | 0.0208 | 0.0266 | 0.0240   | *NaN*   |


"—" = term not present in that period's export; *NaN* = term was filtered out by the stricter `max_df`/`min_df` settings.

**Interpretation:**

- Across all configs, terms like *neural*, *deep*, *graph*, *training*, and *network* show a clear **upward trend** from earlier to later periods, reflecting the rise of deep learning in ML research.
- **more_vocab** gives a larger dictionary (7,778 terms) but slightly lower per-term scores because the weight is spread across more words.
- **stricter_df** gives a smaller dictionary (4,642 terms) and slightly higher scores for the remaining terms, but some common terms like *network* get filtered out entirely.

### 4.2 LDA — Baseline vs Strict Stopwords


| Stopwords  | Period    | Vocab size | Distinctness | Coherence  |
| ---------- | --------- | ---------- | ------------ | ---------- |
| baseline   | 1999–2010 | 702        | 0.8547       | 0.2402     |
| **strict** | 1999–2010 | 670        | **0.9737**   | **0.2604** |
| baseline   | 2011–2015 | 2,058      | 0.8266       | 0.3450     |
| **strict** | 2011–2015 | 2,029      | **0.9481**   | **0.3838** |
| baseline   | 2016–2021 | 5,598      | 0.8480       | 0.3456     |
| **strict** | 2016–2021 | 5,573      | **0.9385**   | **0.4085** |


**Interpretation:** Strict stopwords consistently improve both metrics in every period. By removing generic academic terms, the remaining vocabulary is more informative, so LDA can learn more distinct and coherent topics.

### 4.3 LDA — Hyperparameter Grid Search (strict stopwords only)

We searched over `K ∈ {4, 5, 6, 8, 10}` and `passes ∈ {5, 10, 20, 30}`, selecting the combination that maximised coherence for each period:


| Period    | Best K | Best passes | Vocab size | Distinctness | Coherence |
| --------- | ------ | ----------- | ---------- | ------------ | --------- |
| 1999–2010 | 10     | 5           | 670        | 0.9466       | 0.3414    |
| 2011–2015 | 4      | 30          | 2,029      | 0.9727       | 0.3959    |
| 2016–2021 | 10     | 30          | 5,573      | 0.9775       | 0.5215    |


**Interpretation:**

- The best hyperparameters differ by period — there is no single universal best setting, which highlights the importance of tuning.
- The highest coherence (0.5215) appears in the **2016–2021** period with K=10 and passes=30, reflecting that this larger and more focused corpus supports richer topic structure.
- In **2016–2021**, coherence rises steadily toward larger `K/passes` within the tested grid and peaks at **K=10, passes=30**. In contrast, **2011–2015** peaks at a smaller model (**K=4, passes=30**), indicating period-specific optimal model complexity rather than one globally optimal setting.

### 4.4 BERTopic — Four-variant comparison (beyond outlier ratio)

#### (1) Coverage: outlier ratio


| Variant         | Outlier ratio (topic -1) |
| --------------- | ------------------------ |
| Baseline        | 0.308 (30.8%)            |
| Reduce outliers | 0.000 (0.0%)             |
| KMeans          | 0.000 (0.0%)             |
| MPNet           | 0.174 (17.4%)            |


This confirms a strong coverage trade-off: Baseline is most selective, Reduce-outliers/KMeans assign all documents, and MPNet is intermediate.

#### (2) Topic quality (keyword interpretability from `get_topic`)

Across runs, several themes are consistently meaningful:

- **Neural/deep** (`neural`, `networks`, `training`, `deep`)
- **Reinforcement learning** (`policy`, `reinforcement`, `agent`, `reward`)
- **Graph learning** (`graph`, `gnn`, `node`)
- **Medical/clinical** (`patients`, `clinical`, `disease`)
- **Generative modelling** (`generative`, `gan`, `latent`)

However, MPNet shows one very broad generic topic in early periods (e.g., Topic 0 with terms like `data`, `algorithm`, `clustering`, `classification`) carrying high mass (0.418 in 1999–2010, 0.363 in 2011–2015), which reduces topic specificity despite better embedding quality.

#### (3) Topic structure (granularity and balance)

- **Baseline** keeps more selective clusters but many documents remain unassigned.
- **Reduce outliers** shifts mass into a few dominant topics (e.g., 2016–2021: Topic 0 = 0.156, Topic 1 = 0.112, Topic 2 = 0.099), improving assignment but slightly coarsening structure.
- **KMeans** produces more balanced top-topic shares per period (e.g., 2016–2021 top topics around 0.108/0.107/0.087/0.077/0.061), giving stable and comparable period profiles.
- **MPNet** retains moderate outliers and a broad dominant topic in earlier bins, indicating semantic richness but weaker clustering separation.

#### (4) Temporal signal (period-level mixture shifts)

Your period-wise top-topic outputs show interpretable movement:

- In **later years (2016–2021)**, neural/deep/graph and applied themes (medical, traffic, generative) gain prominence.
- In **earlier periods**, more classical themes (VC/sample complexity, matrix/kernel methods, online regret/bandits, feature/classification topics) are stronger.

So BERTopic supports temporal change, but IDs should be read **within each run** and compared by keyword bundles + period proportions, not by numeric topic matching across models.

### 4.5 HDP (supplementary)

Global HDP period cosine-distance matrix:


|           | 1999–2010 | 2011–2015 | 2016–2021 |
| --------- | --------- | --------- | --------- |
| 1999–2010 | ≈ 0       | 0.0009    | 0.0400    |
| 2011–2015 | 0.0009    | ≈ 0       | 0.0345    |
| 2016–2021 | 0.0400    | 0.0345    | ≈ 0       |


The first two periods are very similar to each other, while **2016–2021** is more distant from both. This is **directionally consistent** with the other methods (all point to a shift in the latest period), but HDP's many low-weight topics limit detailed interpretation.

---

## 5. Evaluation

### (a) Consistency and robustness

- **TF–IDF, LDA, and BERTopic all agree** on the main temporal trend: later periods show stronger deep-learning language (*neural*, *deep*, *graph*, *network*).
- **LDA strict vs baseline**: strict stopwords improve both distinctness and coherence in all three periods (e.g., distinctness: 0.8547 → 0.9737, 0.8266 → 0.9481, 0.8480 → 0.9385; coherence: 0.2402 → 0.2604, 0.3450 → 0.3838, 0.3456 → 0.4085). This is a consistent improvement, not a one-off result.
- **LDA hyperparameter sweep**: the best (K, passes) varies by period, confirming that results are sensitive to these settings and that tuning matters.
- **LDA complexity by period**: later data supports higher-complexity settings (2016–2021 peaks at K=10, passes=30), whereas 2011–2015 peaks at K=4, passes=30; this supports a period-dependent modelling choice rather than a fixed K for all periods.
- **BERTopic**: robustness is supported on four dimensions from your outputs: coverage (0.308/0.000/0.000/0.174), interpretable recurring themes (neural, RL, graph, medical, generative), structure differences (KMeans more balanced; MPNet broader in early bins), and temporal movement (classical earlier vs neural/deep/graph later).
- **HDP**: inter-period distances agree with the "latest bin is different" finding from the other methods.

### (b) Interpretability

- **TF–IDF** is the easiest to interpret at the individual-term level, but does not produce "topics" on its own.
- **LDA** (especially with strict stopwords) produces readable topic–word lists. The strict variant improves separation between topics, making it easier to label them. Note that since each period has its own model, topics must be compared qualitatively (by keywords), not by index number.
- **BERTopic**: Baseline has cleaner core clusters but leaves many documents in `-1`; Reduce-outliers/KMeans improve coverage; MPNet improves semantic richness for some themes but can produce overly broad high-mass topics in early bins.

### (c) Alignment with expected patterns

- The finding that later ML abstracts emphasise *neural*, *deep*, *graph*, and *training* matches well-known trends in the ML literature (the deep learning boom post-2012).
- **Caveat**: the 2016–2021 bin contains the majority of documents, while the 1999–2010 bin has only ~244 documents. Results for the earliest period are therefore noisier and should be interpreted cautiously.

### Integrated interpretation of temporal change

All three methods point in the same direction: the corpus shifts from earlier, more diverse/classical terminology toward later neural/deep/graph-heavy themes. TF–IDF shows this through rising term scores (e.g., *neural* goes from 0.0067 in 2011–2015 to 0.0238 in 2016–2021 under default settings). LDA supports this through improved topic coherence in later periods and through the keywords in its discovered topics. BERTopic's period-level topic mixtures also show compositional shifts over time. The strongest defensible claim is not that "Topic X became Topic Y" (since topic IDs are method-specific), but that the overall language of ML research has shifted toward deep learning terminology over time.

---

## 6. Hyperparameters and Limitations

**What we tuned:**

- LDA: baseline vs strict stopwords; expanded grid search over K ∈ {4, 5, 6, 8, 10} and passes ∈ {5, 10, 20, 30} (strict only).
- BERTopic: four variants (Baseline, Reduce outliers, KMeans, MPNet).
- TF–IDF: small parameter grid (`max_features`, `max_df`, `min_df`).

**Limitations:**

- Topic IDs cannot be matched across methods or across time periods.
- The three time periods have very unequal sizes, which affects the reliability of comparisons.
- BERTopic is computationally expensive and results can vary with embedding model choice.
- HDP produces many low-weight topics, limiting its usefulness beyond a coarse sanity check.

---

## 7. Conclusion

All methods consistently show that later arXiv abstracts place more emphasis on neural/deep/graph-related language, reflecting the evolution of ML research over the past two decades.

**LDA with strict stopwords** is the most robust main model in this study: it improves both distinctness and coherence over the baseline in every period, and performs well across the hyperparameter grid. The strongest single configuration is K=10, passes=30 in the 2016–2021 period (coherence = 0.5215, distinctness = 0.9775).

**BERTopic** adds useful semantic clustering but shows a clear coverage–specificity trade-off (outlier rates range from 0% to 31% depending on the variant), so it is best used as supporting evidence rather than the sole basis for conclusions.

**HDP** is useful as a quick sanity check on temporal structure (it confirms the latest period is most different), but its many overlapping low-weight topics make it less interpretable than the tuned LDA configuration.

---

## Pre-submission Checklist

1. Every number in this report can be traced back to notebook output from a saved run.
2. Evaluation (§5) covers TF–IDF, LDA, and BERTopic under consistency, interpretability, and plausibility.
3. No false claims that topic IDs match across methods or periods.
4. All figures have captions explaining axes and any normalisation.
5. The uneven sizes of the three time periods are acknowledged.
6. Word limit, file naming, and citation requirements are met per the course brief.

