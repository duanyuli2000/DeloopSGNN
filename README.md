# DeleoSGNN: Revisiting Spectral GNNs Through the Lens of Spatial Aggregation

<div align="center">

**DeloopSGNN: Revisiting Spectral GNNs Through the Lens of Spatial Aggregation**

*AAAI Conference on Artificial Intelligence 2026*

**Duanyu Li**¹*, **Huijun Wu**¹*²†, **Min Xie**¹, **Kai Lu**¹, **Wenzhe Zhang**¹, **Zhenwei Wu**¹, **Yong Dong**¹, **Ruibo Wang**¹†

¹ College of Computer Science and Technology, National University of Defense Technology  
* These authors contributed equally.  
† Corresponding author: wuhuijun@nudt.edu.cn, ruibo@nudt.edu.cn

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

</div>

## Abstract

Graph Neural Networks (GNNs) have been studied from two primary perspectives: **spectral** and **spatial**. While spectral GNNs possess notable theoretical advantages rooted in graph signal processing, they often underperform in practice compared to spatial models. 

This work introduces a novel theoretical framework (**S2SMT**) for converting spectral GNNs into the spatial domain, revealing that:
- **Signal looping** and **repeated high-order aggregation** are major causes of over-smoothing in spectral GNNs

Based on this insight, we propose **DeloopSGNN**, which imposes an **acyclic constraint** on the equivalent spatial model, converting deep propagation into deep acyclic propagation.

## Key Contributions

1. **S2SMT Theory**: A rigorous Spectral-to-Spatial Mapping Theorem that establishes equivalence between spectral filters and spatial message passing.

2. **Loop-Free Design**: Novel adjacency matrices that eliminate computational loops while preserving k-hop neighborhood information.

3. **Superior Performance**: Achieves state-of-the-art results on 16 datasets with strong adversarial robustness.

## Method Overview

### Traditional vs. Loop-Free Aggregation

| Method | Aggregation | Loops | Over-smoothing |
|--------|-------------|--------|----------------|
| ChebNet/BernNet | A^k | Yes | Suffers |
| **DeloopSGNN** | A_k^loop-free | No | Mitigated |

### Model Architecture

```
Input Features
    |
    v
[Linear -> ReLU -> Dropout] x 2  (Feature Transformation)
    |
    v
Deloop_prop(K)                    (Loop-Free Propagation)
    |-- theta_0.A_0 + theta_1.A_1 + ... + theta_K.A_K
    |-- Learnable coefficients for each hop
    |
    v
LogSoftmax
    |
    v
Prediction
```

## Requirements

### Software

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Python Dependencies

```bash
# Core deep learning
torch>=2.0.0

# Graph neural networks
torch-geometric>=2.3.0
torch-scatter>=2.1.0
torch-sparse>=0.6.17

# Adversarial robustness
deeprobust>=0.2.1

# Scientific computing
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0
scikit-learn>=1.3.0

# Utilities
pyyaml>=5.4.0
pytorch-lightning>=2.0.0
tqdm>=4.65.0
```

### Installation

```bash
pip install -r requirements.txt
```

## Quick Start

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

## Experimental Results

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

## Project Structure

```
DeloopSGNN/
├── base_models/          # Model implementations
│   ├── deloopsgnn.py   # Main model (with detailed comments)
│   ├── chebnet.py     # Chebyshev networks
│   ├── bernnet.py      # Bernstein networks
│   └── ...
├── data_utils.py        # Data loading utilities
├── exp1_generalization.py  # Standard accuracy experiments
├── exp2_robust.py         # Robustness experiments
├── params.yaml          # Hyperparameter configurations
├── requirements.txt    # Environment dependencies
└── attack_data.zip     # Adversarial attack datasets
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{deloopsgnn2026,
  title={DeloopSGNN: Revisiting Spectral GNNs Through the Lens of Spatial Aggregation},
  author={Li, Duanyu and Wu, Huijun and Xie, Min and Lu, Kai and Zhang, Wenzhe and Wu, Zhenwei and Dong, Yong and Wang, Ruibo},
  booktitle={AAAI Conference on Artificial Intelligence (AAAI)},
  year={2026}
}
```

## Acknowledgments

This work was supported by National University of Defense Technology. We thank the anonymous reviewers for their valuable feedback.

---

<div align="center">

**For questions, please contact: wuhuijun@nudt.edu.cn**

</div>
