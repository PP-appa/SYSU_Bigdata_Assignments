# Task 4：IMDB 电影评论情感分类实验

## 一、实验简介

本实验基于 IMDB 电影评论数据集，实现并对比三种情感分类方法：

- `TF-IDF + Logistic Regression`
- `TF-IDF + LinearSVC`
- `Embedding + LSTM`

实验目标是在统一的数据集划分下，对传统机器学习方法与序列神经网络方法的表现进行对比分析，并输出评估指标、混淆矩阵和训练曲线等结果。

## 二、目录结构

- `train_tfidf_lr.py`：训练并评估 `TF-IDF + Logistic Regression`
- `train_tfidf_svm.py`：训练并评估 `TF-IDF + LinearSVC`
- `train_rnn.py`：训练并评估 `Embedding + LSTM`
- `src/data.py`：IMDB 数据集加载与基础预处理
- `src/features.py`：TF-IDF 特征提取
- `src/evaluate.py`：评估指标计算与混淆矩阵绘图
- `src/utils.py`：JSON 保存等辅助函数
- `outputs/`：保存评估结果、图像和模型参数
- `report/实验报告.md`：实验报告

## 三、环境配置

进入 `task4` 目录后安装依赖：

```bash
cd D:\SYSU_Bigdata_Assignments\task4
pip install -r requirements.txt
```

## 四、运行方式

分别运行以下脚本：

```bash
python train_tfidf_lr.py
python train_tfidf_svm.py
python train_rnn.py
```

## 五、结果输出

实验运行后会在以下目录生成结果文件：

- `outputs/metrics/`：各模型的评估指标 JSON 文件
- `outputs/figures/`：混淆矩阵图与 LSTM 训练曲线图
- `outputs/model/`：LSTM 模型权重文件

## 六、说明

- 数据集通过 `datasets.load_dataset("imdb")` 从 Hugging Face 加载。
- Logistic Regression 与 LinearSVC 使用相同的 TF-IDF 特征，以保证对比公平。
- RNN 模型输入为词 id 序列，而不是 TF-IDF 稀疏向量。
- 实验报告见 `report/实验报告.md`。
