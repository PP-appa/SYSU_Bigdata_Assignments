# Task 7: 混合推荐系统设计

## 任务目标

基于 MovieLens 数据集实现并对比两类推荐方法：

- 协同过滤：基于物品相似度的 ItemCF。
- 基于内容的推荐：使用电影 `genres` 标签构造用户画像。
- 可选改进：将两者按权重融合，观察 Recall@K 是否提升。

## 目录结构

- `run_experiment.py`：运行完整实验并输出指标。
- `src/data.py`：MovieLens 数据下载、读取、留一法划分。
- `src/recommenders.py`：ItemCF、Content-Based、Hybrid 推荐器。
- `src/metrics.py`：Recall@K 评估。
- `outputs/metrics/metrics.json`：实验结果。
- `report/实验报告.md`：报告模板。

## 运行方式

```bash
pip install -r task7/requirements.txt
python task7/run_experiment.py
```

默认会下载 `ml-latest-small`。如果网络不可用，可手动下载并解压到：

```text
task7/data/ml-latest-small/
```

目录中应包含 `ratings.csv` 和 `movies.csv`。
