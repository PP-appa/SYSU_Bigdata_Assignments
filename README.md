# 电影知识图谱

这是一个基于 TMDB 5000 电影数据集构建的小型电影领域知识图谱项目。

工作流程包括：

- 解析电影元数据和演职人员结构化数据
- 使用 spaCy 对电影简介（overview）进行命名实体识别
- 在 Neo4j 中构建和可视化图谱

## 数据集

数据来源：

- TMDB 5000 电影数据集 (TMDB 5000 Movie Dataset)
- https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

主要文件：

- `data/tmdb_5000_movies.csv`
- `data/tmdb_5000_credits.csv`

注意：

- 默认情况下原始数据集不包含在代码仓库中
- 请从 Kaggle 手动下载数据集，并将 CSV 文件放入 `data/` 目录中

## 图谱模式 (Graph Schema)

节点标签 (Node labels)：

- `Movie`
- `Person`
- `Genre`
- `OverviewEntity`

关系类型 (Relationship types)：

- `(:Person)-[:ACTED_IN]->(:Movie)`
- `(:Person)-[:DIRECTED]->(:Movie)`
- `(:Movie)-[:BELONGS_TO]->(:Genre)`
- `(:Movie)-[:MENTIONS]->(:OverviewEntity)`

## 项目结构

- `src/preprocess.py`: 解析原始 CSV 文件并生成图表
- `src/extract_overview_entities.py`: 使用 spaCy 从 `overview` 提取实体
- `src/neo4j_import.cypher`: 将节点和关系导入 Neo4j
- `data/processed/`: 生成的节点表和边表
- `report/实验报告.md`: 实验报告
- `asset/`: 实验报告中使用的截图

## 环境要求

- Python 3.12+
- Neo4j Desktop / Neo4j Browser

安装依赖：

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## 使用方法

1. 生成结构化的节点表和边表：

```bash
python src/preprocess.py
```

2. 使用 spaCy 提取简介中的实体：

```bash
python src/extract_overview_entities.py
```

3. 将 `data/processed/` 目录下生成的 CSV 文件复制到 Neo4j 数据库的 `import` 目录。

4. 运行 `src/neo4j_import.cypher` 中的 Cypher 语句完成导入。

## 代码仓库说明

本代码仓库仅保留源代码、实验报告和截图。

为避免上传过多可复现的大文件，通常不提交以下内容：

- 原始数据集文件
- `data/processed/` 下生成的文件

## 输出结果统计

当前图谱统计：

- `Movie`（电影）: 4803
- `Person`（人物）: 56603
- `Genre`（类型）: 20
- `OverviewEntity`（简介实体）: 9331

Relationships:

- `ACTED_IN`: 106084
- `DIRECTED`: 5166
- `BELONGS_TO`: 12160
- `MENTIONS`: 14487

## Report

The final report is available at:

- `report/实验报告.md`
