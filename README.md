# SYSU Big Data Assignments

这个仓库用于整合和记录我在中山大学（SYSU）大数据课程中的所有作业和项目代码。

## 目录结构

| 任务目录 | 描述说明 | 主要技术栈 |
| :--- | :--- | :--- |
| [**`task1/`**](./task1) | 大数据作业任务一：图像边缘检测与分类对比实验 | Python · OpenCV · PyTorch · Scikit-learn |
| [**`task2/`**](./task2) | 大数据作业任务二：电影知识图谱构建与可视化 | Python · spaCy · Cypher (Neo4j) |

## 各任务简介

### Task 1 — 图像边缘检测与分类
使用 Sobel 算子进行图像边缘检测，并在 CIFAR-10 数据集上对比传统机器学习（SVM、随机森林）与深度学习（MLP、ResNet18）的分类精度与计算效率。

- [实验报告](./task1/实验报告.md)
- [源代码](./task1/src/)

### Task 2 — 电影知识图谱
基于 TMDB 5000 电影数据集，使用 spaCy 命名实体识别提取实体，并在 Neo4j 中构建和可视化电影领域知识图谱。

- [实验报告](./task2/report/实验报告.md)
- [源代码](./task2/src/)

## 完成进度

- [x] Task 1：图像边缘检测与分类对比实验
- [x] Task 2：电影知识图谱构建与可视化
- [ ] 待更新后续作业...

## 环境配置

每个任务目录下均有独立的 `requirements.txt`，请参考各子目录的 `README.md` 进行环境配置。

**Task 1（推荐使用 Conda 管理环境）：**
```bash
conda create -n bigdata_task1 python=3.10
conda activate bigdata_task1
pip install -r task1/requirements.txt
```

**Task 2：**
```bash
pip install -r task2/requirements.txt
python -m spacy download en_core_web_sm
```
