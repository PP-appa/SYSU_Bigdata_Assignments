# SYSU Big Data Assignments

本仓库用于整理和记录中山大学大数据课程的各次作业代码、实验报告和相关结果文件。

## 目录结构

| 任务目录 | 内容说明 | 主要技术 |
| --- | --- | --- |
| [task1/](./task1) | 图像边缘检测与图像分类对比实验 | Python, OpenCV, PyTorch, scikit-learn |
| [task2/](./task2) | 基于 TMDB 数据集的电影知识图谱构建 | Python, spaCy, Neo4j, Cypher |
| [task3/](./task3) | K-means 与 DBSCAN 聚类算法实现与对比 | Python, NumPy, SciPy, scikit-learn, Matplotlib |
| [task4/](./task4) | IMDB 电影评论情感分类实验 | Python, Hugging Face Datasets, scikit-learn, PyTorch |
| [task5/](./task5) | 股票价格预测：ARIMA 与 LSTM 未来 7 日收盘价预测对比 | Python, pandas, statsmodels, PyTorch |
| [task6/](./task6) | 图文多模态问答：基于 ViLT 的视觉问答实验 | Python, PyTorch, Transformers, Pillow |
| [task7/](./task7) | 混合推荐系统设计：协同过滤、内容推荐与加权融合 | Python, pandas, NumPy, scikit-learn |

## 各任务简介

### Task 1

使用 Sobel 算子进行图像边缘检测，并在 CIFAR-10 数据集上对比传统机器学习方法与深度学习方法的分类效果。

- [README](./task1/README.md)
- [Report](./task1/实验报告.md)
- [Source](./task1/src/)

### Task 2

基于 TMDB 5000 电影数据集，提取实体与关系，并在 Neo4j 中构建电影知识图谱。

- [README](./task2/README.md)
- [Report](./task2/report/实验报告.md)
- [Source](./task2/src/)

### Task 3

在 Iris 数据集上实现 K-means 和 DBSCAN，并通过多项聚类指标进行参数扫描和实验分析。

- [README](./task3/README.md)
- [Report](./task3/report/实验报告.md)
- [Source](./task3/src/)

### Task 4

基于 IMDB 电影评论数据集，比较 `TF-IDF + Logistic Regression`、`TF-IDF + LinearSVC` 和 `Embedding + LSTM` 三种情感分类方案。

- [README](./task4/README.md)
- [Report](./task4/report/实验报告.md)
- [Source](./task4/src/)

### Task 5

使用贵州茅台历史股价 CSV 数据，分别使用 `ARIMA` 和 `LSTM` 对未来 `7` 个交易日收盘价进行预测，并通过 `MAE` 和 `RMSE` 对比两种方法的效果。

- [README](./task5/README.md)
- [Report](./task5/report/实验报告.md)
- [Source](./task5/src/)

### Task 6

使用 Hugging Face ViLT 预训练模型完成图文多模态问答任务。输入为图片和自然语言问题，输出模型预测答案，并将问答结果保存为 JSON 文件。

- [README](./task6/README.md)
- [Report](./task6/report/实验报告.md)
- [Source](./task6/src/)

### Task 7

基于 MovieLens `ml-latest-small` 数据集实现推荐系统，对比基于物品相似度的协同过滤、基于电影类型标签的内容推荐，以及加权混合推荐。实验使用 `Recall@10` 对推荐效果进行评价，并分析不同混合权重对结果的影响。

- [README](./task7/README.md)
- [Report](./task7/report/实验报告.md)
- [Source](./task7/src/)
- [Metrics](./task7/outputs/metrics/metrics.json)

## 环境配置

各任务目录均包含独立的 `requirements.txt`，请进入对应目录后按需安装依赖。

### Task 1

```bash
conda create -n bigdata_task1 python=3.10
conda activate bigdata_task1
pip install -r task1/requirements.txt
```

### Task 2

```bash
pip install -r task2/requirements.txt
python -m spacy download en_core_web_sm
```

### Task 3

```bash
pip install -r task3/requirements.txt
python task3/src/main.py
python task3/src/sweep.py
```

### Task 4

```bash
pip install -r task4/requirements.txt
python task4/train_tfidf_lr.py
python task4/train_tfidf_svm.py
python task4/train_rnn.py
```

### Task 5

```bash
pip install -r task5/requirements.txt
python task5/train_arima.py --data-path task5/data/maotai_600519_qfq.csv
python task5/train_lstm.py --data-path task5/data/maotai_600519_qfq.csv
python task5/compare_models.py
```

### Task 6

```bash
pip install -r task6/requirements.txt
python task6/src/infer.py --image task6/examples/Black_dog_retrieving_a_frisbee.jpg --question "What animal is in the image?"
```

### Task 7

```bash
pip install -r task7/requirements.txt
python task7/run_experiment.py
python task7/run_experiment.py --hybrid-alpha 0.3
python task7/run_experiment.py --hybrid-alpha 0.7
```

Task 7 默认会自动下载 MovieLens `ml-latest-small`。如果网络不可用，可手动下载并解压到 `task7/data/ml-latest-small/`，目录中应包含 `ratings.csv` 和 `movies.csv`。
