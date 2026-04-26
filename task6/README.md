# Task 6: 图文多模态问答（VQA）

## 一、作业目标
本作业使用 Hugging Face 提供的 ViLT 预训练模型完成视觉问答任务。输入为一张图片和一个自然语言问题，输出为模型预测答案。

本阶段仅完成项目骨架搭建，后续再补充推理脚本、测试样例和实验报告内容。

## 二、目录结构
- `requirements.txt`：依赖列表
- `examples/`：存放测试图片
- `src/infer.py`：ViLT 推理脚本
- `outputs/answers/`：保存问答结果
- `outputs/figures/`：保存截图或可视化结果
- `report/`：存放实验报告

## 三、计划中的最小可交付
1. 安装依赖并下载预训练模型
2. 准备 1 到 3 张测试图片
3. 编写 `src/infer.py`，支持“图片 + 问题 -> 答案”
4. 截图保存运行结果
5. 补充两页以上简要报告

## 四、当前可用命令
进入项目根目录后安装依赖：

```bash
pip install -r task6/requirements.txt
```

运行单张图片问答：

```bash
python task6/src/infer.py --image task6/examples/test.jpg --question "What animal is in the image?"
```

默认会优先从本地目录 `task6/model/vilt-vqa/` 加载模型；如果该目录不存在，再回退到 Hugging Face。

若使用网络图片：

```bash
python task6/src/infer.py --image https://example.com/test.jpg --question "How many people are there?"
```

结果默认保存到：

```bash
task6/outputs/answers/result.json
```
