# 基于同步-异步功能连接融合的脑疾病识别

## 📌 项目简介
本项目基于fMRI数据，提出一种融合同步与异步功能连接（FBN）的脑疾病识别方法，
结合Transformer与注意力机制，实现对脑区动态连接关系的建模。

## 🚀 方法特点
- 同步功能连接（S-FBN）
- 异步功能连接（A-FBN）
- 时间分割机制（Temporal Segment）
- 基于RBF核的注意力机制
- 稀疏化脑网络建模

## 🧠 模型结构
fMRI → 时间分割 → 异步建模 → 注意力 → 分类

## 📂 项目结构
model/ 模型结构
data/ 数据说明
main.py 训练入口

## ⚙️ 运行方式
```bash
pip install torch numpy
python main.py

📊 数据说明
使用ADNI数据集（未公开上传）
