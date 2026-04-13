# Task 5：股票价格预测

## 一、实验简介

本实验选择某只股票的历史日线数据（CSV 格式），分别使用 `ARIMA` 与 `LSTM` 两种时间序列方法预测未来 `7` 个交易日的收盘价，并使用 `MAE` 与 `RMSE` 指标对两种模型进行对比分析。

当前实现默认使用：

- 预测目标：收盘价（若存在 `Adj Close` 则优先使用复权收盘价）
- 测试集：最后 `7` 个交易日
- 训练集：除最后 `7` 个交易日之外的全部历史数据

## 二、目录结构

- `download_data.py`：从公开数据源下载股票日线 CSV
- `train_arima.py`：训练并评估 `ARIMA`
- `train_lstm.py`：训练并评估 `LSTM`
- `compare_models.py`：汇总两种模型的 `MAE/RMSE`
- `src/data.py`：股票数据读取、划分与归一化
- `src/evaluate.py`：回归指标计算与预测图绘制
- `src/utils.py`：JSON 保存、随机种子等辅助函数
- `outputs/`：保存实验指标、图像、预测结果和模型参数
- `report/实验报告.md`：实验报告

## 三、环境配置

进入 `task5` 目录后安装依赖：

```bash
cd D:\SYSU_Bigdata_Assignments\task5
pip install -r requirements.txt
```

## 四、数据准备

### 方案 A：使用下载脚本

```bash
python download_data.py --symbol aapl.us --output data/aapl_us_daily.csv
```

说明：

- `aapl.us` 表示苹果公司股票，可替换为其他 `Stooq` 支持的日线代码
- 脚本会下载包含 `Date/Open/High/Low/Close/Volume` 的 CSV 文件

### 方案 B：使用你自己的 CSV

将你的股票历史数据放到 `task5/data/` 中，然后在训练命令里通过 `--data-path` 指定。CSV 至少需要：

- 日期列：`Date` / `date`
- 价格列：`Adj Close` / `Close`

## 五、运行方式

### 1. 训练并评估 ARIMA

```bash
python train_arima.py --data-path data/aapl_us_daily.csv
```

### 2. 训练并评估 LSTM

```bash
python train_lstm.py --data-path data/aapl_us_daily.csv
```

### 3. 汇总模型对比结果

```bash
python compare_models.py
```

## 六、结果输出

实验完成后，默认会生成以下文件：

- `outputs/metrics/arima_metrics.json`
- `outputs/metrics/lstm_metrics.json`
- `outputs/metrics/model_comparison.csv`
- `outputs/forecasts/arima_forecast.csv`
- `outputs/forecasts/lstm_forecast.csv`
- `outputs/figures/arima_forecast.png`
- `outputs/figures/lstm_forecast.png`
- `outputs/figures/lstm_training_curve.png`
- `outputs/model/lstm_state_dict.pt`

## 七、实验说明

- `ARIMA` 使用训练集价格序列进行参数搜索，按 `AIC` 选择最优 `(p, d, q)`。
- `LSTM` 使用滑动窗口构造监督学习样本，训练完成后从训练集末尾递归预测未来 `7` 步。
- 两种模型都只在训练集上拟合，最后与测试集的最后 `7` 个真实价格进行比较。
- 若更换股票或调整参数，只需修改命令行参数即可重新实验。
