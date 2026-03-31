# 作业3：聚类算法对比实验（Iris）

本任务包含：
- 手写实现 `KMeans` 核心逻辑
- 手写实现 `DBSCAN` 核心逻辑
- 在 Iris 数据集上做参数对比
- 使用准确率、轮廓系数、Calinski-Harabasz 指数评估结果

## 目录结构

```text
task3/
├── src/
│   ├── kmeans.py
│   ├── dbscan.py
│   ├── metrics_custom.py
│   └── main.py
├── output/                 # 运行后自动生成
├── requirements.txt
└── README.md
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行

```bash
python src/main.py
```

可调参数示例：

```bash
python src/main.py --k 2 --eps 0.5 --min_samples 4
python src/main.py --k 4 --eps 0.8 --min_samples 8
```

运行后将输出：
- `output/metrics.json`
- `output/kmeans_scatter.png`
- `output/dbscan_scatter.png`

## 参数批量实验

```bash
python src/sweep.py
```

将生成：
- `output/kmeans_sweep.csv`
- `output/dbscan_sweep.csv`
