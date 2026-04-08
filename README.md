# SYSU Big Data Assignments

本仓库用于整理和记录中山大学大数据课程的各次作业代码、实验报告和相关结果文件。

## 目录结构

| 任务目录 | 内容说明 | 主要技术 |
| --- | --- | --- |
| [task1/](./task1) | 图像边缘检测与图像分类对比实验 | Python, OpenCV, PyTorch, scikit-learn |
| [task2/](./task2) | 基于 TMDB 数据集的电影知识图谱构建 | Python, spaCy, Neo4j, Cypher |
| [task3/](./task3) | K-means 与 DBSCAN 聚类算法实现与对比 | Python, NumPy, SciPy, scikit-learn, Matplotlib |
| [task4/](./task4) | IMDB 电影评论情感分类实验 | Python, Hugging Face Datasets, scikit-learn, PyTorch |

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
