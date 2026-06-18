# CNN Artist Classification

使用卷積神經網路（CNN）對畫作進行**畫家分類**的深度學習作業（HW1）。模型讀入一張畫作圖片，預測其作者屬於哪一位畫家。

資料集為 [Best Artworks of All Time](https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time)，包含 50 位知名畫家的畫作。

## 專案內容

| 檔案 | 說明 |
| --- | --- |
| `HW1_ARTCNN_110401544 (1).ipynb` | 主要筆記本：資料前處理、建模、訓練、評估與預測 |
| `train/artists.csv` | 畫家清單與每位畫家的畫作數量等中繼資料 |

> 圖片資料集（`train/train_resized/`、`test_resized/`）因檔案過大未納入版控，請另行下載並放置對應路徑。

## 方法流程

1. **資料前處理**：讀取畫家清單，建立「畫家名稱 ↔ 類別索引」映射；從檔名解析 label。
2. **建立 Dataset**：以 `tf.data` 載入圖片，resize 成 224×224 並正規化到 `[0, 1]`，與 one-hot label 配對後 shuffle。
3. **切分資料**：80% 訓練 / 20% 驗證，batch size = 32。
4. **模型架構**：4 層 `Conv2D + BatchNormalization + MaxPooling`（32 → 64 → 128 → 256 filters），接全連接層輸出各畫家類別的 softmax 機率。
5. **訓練**：Adam optimizer（初始 lr = 0.001）、`categorical_crossentropy`，搭配 `ReduceLROnPlateau` 自動調整學習率，訓練 20 個 epoch。
6. **評估與預測**：在測試集計算 loss / accuracy，並提供 `predict_author()` 對單張圖片預測畫家，支援上傳圖片即時辨識。

## 環境需求

- Python 3
- TensorFlow / Keras
- NumPy、Pandas、OpenCV (`cv2`)、Matplotlib、Seaborn

原始開發環境為 Google Colab（使用 Google Drive 掛載資料集）。

## 使用方式

1. 下載資料集，放到筆記本中設定的 `train_dir`、`test_dir` 路徑。
2. 開啟 `HW1_ARTCNN_110401544 (1).ipynb`，由上而下依序執行各個 cell。

## 資料不平衡

各畫家的畫作數量差異很大（詳見筆記本中的長條圖），此類別不平衡會影響模型訓練表現，是後續可改進的方向之一。
