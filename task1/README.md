# 大数据作业一：图像边缘检测与分类对比实验项目

此项目为验证传统计算机视觉算法（Sobel边缘检测）、经典机器学习（SVM、随机森林）与深度学习模型（MLP、ResNet18）在图像特征提取及分类任务上的差异性表现。

## 项目结构
```text
task1/
├── data/
│   └── image.jpg                   # Sobel测试使用的源图片
├── output/
│   ├── model_results.json          # 各类模型训练精确度与时间原始数据
│   ├── performance_comparison.png  # 模型性能图表（自动生成）
│   └── sobel_result.png            # 边缘检测输出示例图
├── src/
│   ├── sobel_edge.py               # Sobel边缘检测试验代码
│   ├── traditional_ml_full.py      # SVM与随机森林（带PCA降维）实现代码
│   ├── mlp_gpu.py                  # PyTorch版多层感知机（GPU加速）
│   ├── resnet_classifier.py        # ResNet18深度卷积网络学习
│   └── plot_results.py             # 实验数据结果图表渲染器
├── prompt.md                       # 实验设计需求文档
├── 实验报告.md                     # 最终生成的实验总结与图表分析（核心提交）
└── requirements.txt                # 项目依赖清单
```

## 环境安装与运行指南

本项目使用了 Conda 进行环境管理，解决由于 C++ 底层动态链接库(`c10.dll`)在 Windows 下引起的崩溃问题。

1. **创建环境并激活**
```bash
conda create -n bigdata_task python=3.10
conda activate bigdata_task
```
2. **安装依赖**
```bash
pip install -r requirements.txt
# 或者针对具体GPU配置安装 PyTorch（如果 pip 版本无法调用 CUDA）：
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

3. **执行代码**
```bash
# 运行分类器与统计模块
python src/sobel_edge.py
python src/traditional_ml_full.py
python src/mlp_gpu.py
python src/resnet_classifier.py
python src/plot_results.py
```
