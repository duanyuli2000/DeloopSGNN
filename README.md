# DeleoSGNN: Revisiting Spectral GNNs Through the Lens of Spatial Aggregation

<div align="center">

**DeloopSGNN: 从空域聚合视角重新审视谱域图神经网络**

*AAAI Conference on Artificial Intelligence 2026*

**Duanyu Li**¹*, **Huijun Wu**¹*²†, **Min Xie**¹, **Kai Lu**¹, **Wenzhe Zhang**¹, **Zhenwei Wu**¹, **Yong Dong**¹, **Ruibo Wang**¹†

¹ College of Computer Science and Technology, National University of Defense Technology  
* These authors contributed equally.  
† Corresponding author: wuhuijun@nudt.edu.cn, ruibo@nudt.edu.cn

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

</div>

## 📌 Abstract

Graph Neural Networks (GNNs) have been studied from two primary perspectives: **spectral** and **spatial**. While spectral GNNs possess notable theoretical advantages rooted in graph signal processing, they often underperform in practice compared to spatial models. 

This work introduces a novel theoretical framework (**S2SMT**) for converting spectral GNNs into the spatial domain, revealing that:
- **Signal looping** and **repeated high-order aggregation** are major causes of over-smoothing in spectral GNNs

Based on this insight, we propose **DeloopSGNN**, which imposes an **acyclic constraint** on the equivalent spatial model, converting deep propagation into deep acyclic propagation.

## 🔑 Key Contributions

1. **S2SMT Theory**: A rigorous Spectral-to-Spatial Mapping Theorem that establishes equivalence between spectral filters and spatial message passing.

2. **Loop-Free Design**: Novel adjacency matrices that eliminate computational loops while preserving k-hop neighborhood information.

3. **Superior Performance**: Achieves state-of-the-art results on 16 datasets with strong adversarial robustness.

## 🏗️ Method Overview

### Traditional vs. Loop-Free Aggregation

| Method | Aggregation | Loops | Over-smoothing |
|--------|-------------|--------|----------------|
| ChebNet/BernNet | Ã^k | ✓ Yes | ✗ Suffers |
| **DeloopSGNN** | Ã_k^loop-free | ✗ No | ✓ Mitigated |

### Model Architecture

```
Input Features
    ↓
[Linear → ReLU → Dropout] × 2  (Feature Transformation)
    ↓
Deloop_prop(K)                    (Loop-Free Propagation)
    ├─ θ₀·Ã₀ + θ₁·Ã₁ + ... + θ_K·Ã_K
    └─ Learnable coefficients for each hop
    ↓
LogSoftmax
    ↓
Prediction
```

## 🖥️ Experimental Environment

### 硬件环境

| 配置项 | 详情 |
|--------|------|
| 服务器 | SSH 服务器 (lab-server-A6000-remote) |
| IP 地址 | 120.46.131.227 |
| 用户 | hello |
| GPU | NVIDIA GPU (如有) |

### 软件环境

#### 操作系统
- Linux (Ubuntu/Debian)

#### Python 环境
```bash
# 核心依赖
torch>=2.0.0          # PyTorch 深度学习框架
torch-geometric>=2.3.0  # 图神经网络框架
torch-scatter>=2.1.0    # 稀疏矩阵运算
torch-sparse>=0.6.17    # 稀疏矩阵支持

# 对抗鲁棒性
deeprobust>=0.2.1       # 图对抗防御工具包

# 科学计算
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0
scikit-learn>=1.3.0

# 辅助工具
pyyaml>=5.4.0
pytorch-lightning>=2.0.0
tqdm>=4.65.0
```

#### 安装方式

```bash
# 方式1: 使用 requirements.txt
pip install -r requirements.txt

# 方式2: 手动安装 PyG 相关包
pip install torch torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.10.0+cpu.html
# (根据 CUDA 版本选择对应的 whl 文件)
```

#### 服务器实际环境 (参考)
```
torch              2.10.0
numpy              1.26.4
pandas             2.2.3
scipy              1.15.3
PyYAML             5.4.1
```

### 工作目录结构

```
/home/hello/ldy/liclaw/
├── workspace/          # 代码工作区
│   └── DeleopSGNN/    # 项目代码
├── data/              # 小型数据集
├── outputs/           # 实验输出
└── temp/              # 临时文件
```

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/duanyuli2000/DeloopSGNN.git
cd DeleopSGNN
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Experiments

**Standard Task (Accuracy):**

```bash
python exp1_generalization.py --model DeloopSGNN --dataset cora --n_trials 3
```

**Adversarial Robustness:**

```bash
python exp2_robust.py --model DeloopSGNN --dataset cora --attack metattack
```

### Available Datasets

- Homophilic: Cora, Citeseer, Pubmed, Reddit, ogbn-arxiv
- Heterophilic: Chameleon, Squirrel, Film, Texas, Wisconsin, Cornell

### Available Models

| Model | Description |
|-------|-------------|
| `DeloopSGNN` | Proposed method with loop-free propagation |
| `ChebNet` | Chebyshev polynomial filters |
| `BernNet` | Bernstein polynomial filters |
| `JacobiConv` | Jacobi polynomial filters |
| `GPRGNN` | Generalized PageRank GNN |
| `GCNSVD` | Low-rank approximation |

## 📊 Experimental Results

### Accuracy Comparison

| Dataset | DeloopSGNN | ChebNet | BernNet | GPRGNN |
|---------|-------------|---------|---------|---------|
| Cora | **84.5%** | 82.2% | 78.8% | 76.0% |
| Citeseer | **77.3%** | 73.1% | 71.5% | 69.2% |
| Chameleon | **68.7%** | 62.3% | 58.9% | 55.4% |

### Key Findings

1. **Loop-free design consistently improves performance** across all spectral baselines
2. **Adversarial robustness** significantly enhanced by removing short loops
3. **Generalization** improved on both homophilic and heterophilic graphs

## 📁 Project Structure

```
DeloopSGNN/
├── base_models/          # 模型实现
│   ├── deloopsgnn.py   # 主模型 (含详细注释)
│   ├── chebnet.py     # Chebyshev 网络
│   ├── bernnet.py      # Bernstein 网络
│   └── ...
├── data_utils.py        # 数据加载工具
├── exp1_generalization.py  # 标准准确率实验
├── exp2_robust.py         # 鲁棒性实验
├── params.yaml          # 超参数配置
├── requirements.txt    # 环境依赖
└── attack_data.zip     # 对抗攻击数据集
```

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@article{deloopsgnn2026,
  title={DeloopSGNN: Revisiting Spectral GNNs Through the Lens of Spatial Aggregation},
  author={Li, Duanyu and Wu, Huijun and Xie, Min and Lu, Kai and Zhang, Wenzhe and Wu, Zhenwei and Dong, Yong and Wang, Ruibo},
  booktitle={AAAI Conference on Artificial Intelligence (AAAI)},
  year={2026}
}
```

## 🙏 Acknowledgments

This work was supported by National University of Defense Technology. We thank the anonymous reviewers for their valuable feedback.

---

<div align="center">

**For questions, please contact: wuhuijun@nudt.edu.cn**

</div>
