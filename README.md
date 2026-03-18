# QuaDRiGa 2×2 MIMO 通道生成與 WGAN-GP 建模

本專案包含兩個主要程式：

- `quadriga_mimo_2x2_generator.m`
- `mimo_channel_wgan_gp.py`
- 
## 檔案說明

### 1. `quadriga_mimo_2x2_generator.m`

此 MATLAB 程式負責：

- 建立 QuaDRiGa 模擬環境
- 設定 2×2 MIMO 天線陣列
- 建立基地台與使用者幾何位置
- 設定 UE 移動軌跡
- 產生 3GPP 38.901 UMi NLOS 場景下的通道係數
- 將多路徑通道加總成 flat-fading 通道矩陣
- 將每個複數 2×2 通道矩陣轉為實數向量
- 輸出 `.mat` 資料檔供 Python 端訓練使用

輸出檔案：

- `quadriga_mimo_2x2_dataset.mat`

主要輸出變數：

- `H_coeff`：`[Nr, Nt, Npaths, Nsnapshots]`，原始多路徑通道係數
- `H_flat`：`[Nr, Nt, Nsnapshots]`，將 path 維度加總後的 flat-fading 通道
- `H_vec`：`[Nsnapshots, 2*Nr*Nt]`，實部與虛部交錯排列後的向量化特徵

---

### 2. `mimo_channel_wgan_gp.py`

此 Python 程式負責：

- 讀取 MATLAB 產生的通道資料
- 若需要，自動將複數通道矩陣轉為實數向量
- 對資料做標準化處理
- 建立 Generator 與 Critic
- 使用 WGAN-GP 進行訓練
- 產生新的通道樣本
- 輸出模型、圖表與統計摘要

主要輸入資料變數支援：

- `H_vec`
- `H_flat`
- `H_mimo`

輸出資料夾預設為：

- `mimo_gan_results/`

其中包含：

- `mimo_wgan_gp.pt`：訓練完成的 Generator 權重與標準化參數
- `generated_mimo_channels.mat`：生成的通道樣本
- `summary.txt`：測試集與生成資料的統計比較
- `training_curve.png`：訓練 loss 曲線
- `real_vs_fake_H11.png`：真實與生成的 `H11` 複平面散佈比較圖


### 2. 執行通道生成程式

在 MATLAB 中執行：

```matlab
quadriga_mimo_2x2_generator
```

執行成功後，會在目前資料夾產生：

```text
quadriga_mimo_2x2_dataset.mat
```

---

## Python 端使用方式

當 MATLAB 端完成資料生成後，即可在 Python 端執行：

```bash
python mimo_channel_wgan_gp.py --data quadriga_mimo_2x2_dataset.mat --epochs 300
```

常用參數：

- `--data`：輸入 `.mat` 資料檔
- `--out`：輸出資料夾名稱
- `--epochs`：訓練回合數
- `--batch_size`：batch 大小
- `--latent_dim`：Generator 的 latent vector 維度
- `--seed`：隨機種子

---

## 程式流程說明

### `quadriga_mimo_2x2_generator.m` 的主要流程

1. 設定中心頻率、天線數、高度、移動距離與速度
2. 建立 QuaDRiGa 模擬參數物件 `qd_simulation_parameters`
3. 建立 `qd_layout`
4. 配置 Tx 與 Rx 的 2 元件 ULA 陣列
5. 設定基地台與 UE 的位置與移動軌跡
6. 指定場景為 `3GPP_38.901_UMi_NLOS`
7. 產生原始通道係數 `H_coeff`
8. 將所有 path 加總得到 `H_flat`
9. 將每個 2×2 複數通道矩陣轉為 8 維實數向量 `H_vec`
10. 存成 `.mat` 檔

對 2×2 MIMO 而言，每個 snapshot 的向量化格式如下：

```text
[Re(H11), Im(H11), Re(H21), Im(H21), Re(H12), Im(H12), Re(H22), Im(H22)]
```

---

### `mimo_channel_wgan_gp.py` 的主要流程

1. 讀取 `.mat` 檔
2. 若資料為複數矩陣，則轉換為實數向量特徵
3. 對每一維做標準化
4. 建立 Generator 與 Critic
5. 使用 WGAN-GP 訓練生成模型
6. 生成與測試集同數量的假通道樣本
7. 計算真實資料與生成資料的統計差異
8. 輸出模型、圖與摘要檔案

---

## 模型架構簡述

### Generator

Generator 是一個多層全連接網路，輸入為隨機 latent vector，輸出為與通道特徵維度相同的向量。

若為 2×2 MIMO，則輸出維度為：

```text
2 × Nr × Nt = 2 × 2 × 2 = 8
```

也就是一個 8 維實數向量，對應到一個通道 snapshot。

### Critic

Critic 也是一個多層全連接網路，負責輸入真實或生成的通道向量，輸出一個分數。

在 WGAN 中，Critic 並不是輸出機率，而是學習真實分布與生成分布之間的 Wasserstein 距離近似。

### Gradient Penalty

本程式採用 WGAN-GP，而非原始 GAN。

其優點是訓練較穩定，不需要使用 weight clipping。程式中透過在真樣本與假樣本之間做插值，限制 Critic 的梯度範數接近 1，以滿足 Lipschitz 條件。

---

## 資料格式說明

### 原始複數通道矩陣

```text
H_flat: [Nr, Nt, Nsnapshots]
```

例如：

```text
[2, 2, 1000]
```

表示共有 1000 個 2×2 複數通道矩陣。

### 向量化後特徵

```text
H_vec: [Nsnapshots, 2*Nr*Nt]
```

對 2×2 MIMO 而言：

```text
[1000, 8]
```

這就是 GAN 實際訓練時使用的資料格式。

---

## 輸出結果解讀

### 1. `training_curve.png`

顯示 Critic 與 Generator 在訓練過程中的 loss 變化。

GAN 的 loss 不一定會像監督式學習一樣單調下降，因此應搭配其他結果一併判讀。

### 2. `real_vs_fake_H11.png`

比較真實資料與生成資料中 `H11` 的複平面散佈圖。

若兩者的分布外觀接近，表示生成器已能一定程度模仿通道分布。

### 3. `summary.txt`

包含兩個基本統計量：

- `mean_error_l2`：真實資料與生成資料平均值之 L2 誤差
- `covariance_error_fro`：真實資料與生成資料共變異數矩陣之 Frobenius 誤差

這兩個值越小，代表生成資料與真實資料在整體統計特性上越接近。






