# 📚 Awesome Time Series Generation

> 个人文献调研清单：专注于时间序列生成、扩散模型与基础模型。
> 最后更新时间: 2026-01-20

## 目录 (Table of Contents)

- [🚀 时间序列基础模型 (Foundation Models)](#-时间序列基础模型-foundation-models)
- [🌊 基于扩散模型的生成 (Diffusion-based Generation)](#-基于扩散模型的生成-diffusion-based-generation)
- [🗣️ 文本驱动与可控生成 (Text-to-Series & Controllable)](#️-文本驱动与可控生成-text-to-series--controllable)
- [🛠️ 通用生成与分解 (General Synthesis & Decomposition)](#️-通用生成与分解-general-synthesis--decomposition)

---

## 🚀 时间序列基础模型 (Foundation Models)
> 旨在构建通用的、跨域的时间序列“大模型”。

| 📅 年份 | 📑 标题 (Title) | 🏛 会议 | 💡 核心点/备注 | 🔗 资源 |
| :--- | :--- | :--- | :--- | :--- |
| 2024 | **MOMENT: A Family of Open Time-series Foundation Models** | ICML | 开源时间序列基础模型家族 | [Code](https://github.com/moment-timeseries-foundation-model/moment) |
| 2024 | **Chronos: Learning the Language of Time Series** | ICML | 将时序视为语言，基于Transformer的预测模型 | [Code](https://github.com/amazon-science/chronos-forecasting) |
| 2025 | **Lag-Llama: Towards Foundation Models for Probabilistic...** | ICLR | 概率时间序列预测的基础模型 | [Code](https://github.com/time-series-foundation-models/lag-llama) |
| 2024 | UniTS: Building a Unified Time Series Model | NeurIPS | 统一多任务时间序列模型 | [Code](https://github.com/mims-harvard/UniTS) |

## 🌊 基于扩散模型的生成 (Diffusion-based Generation)
> 利用 Diffusion Model 进行高质量、可解释的时间序列生成。

| 📅 年份 | 📑 标题 (Title) | 🏛 会议 | 💡 核心点/备注 | 🔗 资源 |
| :--- | :--- | :--- | :--- | :--- |
| 2024 | **Diffusion-TS: Interpretable Diffusion for General Time Series** | ICLR | 提供可解释性的通用扩散生成框架 | [Code](https://github.com/Y-debug-sys/Diffusion-TS) |
| 2024 | FIDE: Frequency-Inflated Conditional Diffusion Model... | NeurIPS | 频率增强条件扩散，针对极端感知生成 | [Code](https://github.com/galib19/FIDE) |
| 2025 | Ctrl-Adapter: An Efficient and Versatile Framework... | ICLR | 适配任意扩散模型的高效控制框架 | [Code](https://arxiv.org/pdf/2404.09967) |
| 2025 | Diffusion Transformers for Tabular Data Time Series Generation | ICLR | 表格数据时间序列生成的 DiT 应用 | [Code](https://github.com/fabriziogaruti/TabDiT) |
| 2025 | Population Aware Diffusion for Time Series Generation | AAAI | 群体统计特征感知的生成 | [Code](https://github.com/wmd3i/PaD-TS) |
| 2024 | TimeLDM: Latent Diffusion Model for Unconditional... | Preprint | 效率优化与长序列生成（潜在扩散模型） | [Link](https://arxiv.org/abs/2407.04211) |

## 🗣️ 文本驱动与可控生成 (Text-to-Series & Controllable)
> 通过文本描述或特定指令来生成/编辑时间序列。

| 📅 年份 | 📑 标题 (Title) | 🏛 会议 | 💡 核心点/备注 | 🔗 资源 |
| :--- | :--- | :--- | :--- | :--- |
| 2025 | **VerbaTS: Generating Time Series from Texts** | ICML/NeurIPS | 文本交互生成时间序列 | [Code](https://github.com/seqml/VerbaTS) |
| 2025 | **T2S: High-resolution Time Series Generation with Text...** | IJCAI | 文本到序列扩散模型，高分辨率生成 | [Code](https://github.com/WinfredGc/T2S) |
| 2025 | TimeDP: Learning to Generate Multi-Domain Time Series... | AAAI | 利用领域提示词生成多域时序 | [Code](https://arxiv.org/abs/2501.05403) |
| 2024 | Towards Editing Time Series | NeurIPS | 时间序列编辑（局部修改趋势） | [Code](https://github.com/seqml/TEdit) |

## 🛠️ 通用生成与分解 (General Synthesis & Decomposition)
> 关注数据分解、稀缺性问题及通用生成技术。

| 📅 年份 | 📑 标题 (Title) | 🏛 会议 | 💡 核心点/备注 | 🔗 资源 |
| :--- | :--- | :--- | :--- | :--- |
| 2025 | Effective Series Decomposition and Components Learning... | ICDM | 周期性、趋势性扩散模型（未开源） | [Link](链接) |
| 2024 | Generative Time Series Forecasting with Diffusion... | NeurIPS | 结合分解、去噪、解缠的生成式预测 | [Code](https://github.com/PaddlePaddle/PaddleSpatial) |
| 2025 | Time Series Generation Under Data Scarcity... | NeurIPS | 数据稀缺下的统一生成建模方法 | [Code](https://arxiv.org/abs/2505.20446) |

---
*Generated based on user research list.*
