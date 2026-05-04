# ai-coursework-group13
# ai-coursework-group13
My primary responsibility in this project was the design and implementation of the main text-analysis pipeline for Task 3: Comparative Corpus Analysis and Topic Evolution. I developed the core experimental framework used to analyse temporal changes in arXiv abstracts across the 1999-2021 period.

My specific contributions included:

• Developing the preprocessing strategy, including separate cleaned-text representations for TF-IDF/LDA/HDP models and embedding-based BERTopic.

• Implementing year extraction from arXiv IDs and dividing the corpus into three time periods: 1999-2010, 2011-2015, and 2016-2021.

• Building the TF-IDF analysis module, including unigram TF-IDF, n-gram TF-IDF, distinctive-term analysis, and cross-period cosine-distance comparison.

• Implementing the main topic-modelling pipelines, including period-specific unigram and bigram LDA, global LDA for shared-space comparison, BERTopic configuration comparisons, and global/period-specific HDP as a supplementary robustness check.

• Applying shared-space comparison logic to LDA and BERTopic so that topic distributions could be compared across time periods more consistently.

• Creating and managing the GitHub repository, organising task allocation, coordinating the division of work, and checking final code submissions for consistency with the overall research design.

• Contributing to report writing by providing detailed code explanation documents, method descriptions, and presentation of experimental results.
