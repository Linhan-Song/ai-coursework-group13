# EMATM0067 — Task 3：比較式語料分析（ML arXiv 摘要）

本 repo 為 **Introduction to AI and Text Analytics** 小組作業之 **Task 3**：在機器學習相關 arXiv 摘要上，比較不同文字分析方法如何呈現**跨時段**的研究焦點變化。

- **課程**：EMATM0067（University of Bristol）  
- **主要分支（本組／本人工作彙整）**：[bryan_branch](https://github.com/Linhan-Song/ai-coursework-group13/tree/bryan_branch)

---

## 專案檔案說明

| 檔案 | 說明 |
|------|------|
| **`Bryan's_AI_task3_updated.ipynb`** | 主要分析 notebook：前處理、TF-IDF／LDA／BERTopic（與輔助 HDP）實驗、圖表與指標彙整。 |
| **`Task3_report_draft.md`** | 小組報告草稿／文字素材（與 notebook 結果互相對照）。 |
| **`README.md`** | 本說明檔。 |

---

## Bo-Yan Lu（本人）在專案中的角色

後期工作重心放在 **參數調整（hyperparameter tuning）** 與 **依指標選出較佳設定**，讓比較不是「只跑一次模型」，而是有清楚依據、可重現的實驗設計。

### 1. LDA：停用詞與主題數／訓練輪數

- **Baseline vs strict stopwords**：在相同流程下比較兩套停用詞，觀察 **coherence（一致性）** 與 **distinctness（主題區辨度）**，確認較嚴格的停用詞在三期是否**穩定改善**主題品質。  
- **主題數 `K` 與 `passes` 網格**：在不同時段語料上搜尋較佳組合，並以 coherence 等指標挑出 **各期「相對最好」的設定**（需注意：**最佳 K 會隨時段與語料規模改變**，因此不強迫三期共用同一組固定參數）。

### 2. BERTopic：變體與叢集／嵌入相關設定

- 比較多種實作變體（例如 **Baseline HDBSCAN**、**reduce outliers**、**KMeans**、**不同 sentence embedding** 等），並以 **outlier 比例（topic −1）**、主題可讀性與跨期敘事一致性，說明 **覆蓋率 vs 特異性** 的取捨。  
- 目的：在「語意題目較清楚」與「文件被指派到題目的比例」之間，選出**較適合寫進報告結論**的設定與對照方式。

### 3. TF-IDF：詞彙表與文件頻率門檻

- 針對 `max_features`、`min_df`、`max_df` 等設定做對照，觀察代表詞／代表片語與跨期相似度是否穩定，並避免「過度過濾導致重要詞被洗掉」或「詞彙過大導致訊號被稀釋」。

### 4. 跨方法「選優」的共同原則

- **先固定可比較的評估框架**（同一時段切分、同一套前處理邏輯對應到各方法假設），再在各自方法內做參數搜尋。  
- **以指標與質性閱讀並行**：數值上挑較佳設定，但仍回到「關鍵詞／主題詞是否可解釋、是否與已知 ML 發展敘事一致」做最後取捨。

---

## 使用方式（簡要）

1. 建議使用課程提供或 repo 內的 **conda／venv** 環境（依 notebook 開頭的 `import` 與安裝區塊為準）。  
2. 在 Jupyter／VS Code／Cursor 開啟 **`Bryan's_AI_task3_updated.ipynb`**，依序執行；資料若來自 Hugging Face，首次下載需網路連線。  
3. 大型輸出（圖表、矩陣）建議以 **輸出資料夾** 或報告附錄整理，避免單一 notebook 過重。

---

## 引用與學術誠實

- 小組報告與個人反思請遵守課程對 **引用、抄襲與可生成內容** 的規定。  
- 本 README 僅概述分工與實驗取向，**不取代**正式報告中的方法細節與完整結果表。

---

## 聯絡

若為同組成員：請以組內約定管道（例：GitHub Issues／課程討論區）聯繫；外部人士請尊重課程資料與授權範圍。
