# 📚 Awesome Time Series Generation

> 个人文献调研清单：专注于时间序列生成、扩散模型与基础模型。
> 最后更新时间: 2026-01-21

## 目录 (Table of Contents)

- [ 时间序列基础模型 (Foundation Models)](#-时间序列基础模型-foundation-models)
- [自回归模型的时序生成 (Autoregressive Modeling)](#-自回归模型的时序生成-autoregressive-modeling)
- [ 基于扩散模型的生成 (Diffusion-based Generation)](#-基于扩散模型的生成-diffusion-based-generation)
- [ 文本驱动与可控生成 (Text-to-Series & Controllable)](#️-文本驱动与可控生成-text-to-series--controllable)
- [ 通用生成与分解 (General Synthesis & Decomposition)](#️-通用生成与分解-general-synthesis--decomposition)

---

##  时间序列基础模型 (Foundation Models)
> 旨在构建通用的、跨域的时间序列“大模型”。

|  年份 |  标题 (Title) | 会议/期刊 |  核心点/备注 |  资源 |
| :--- | :--- | :--- | :--- | :--- |
| **2025** | **MetaIndux-TS: Frequency-Aware AIGC Foundation Model...** | IEEE TNNLS | **[工业基础模型]** 频域感知的扩散基础模型，具备工业场景 Zero-shot 生成能力 | [Code](https://github.com/Dolphin-wang/MetaIndux) |
| 2024 | **MOMENT: A Family of Open Time-series Foundation Models** | ICML | 开源时间序列基础模型家族 | [PDF](https://arxiv.org/pdf/2402.03885) \| [Code](https://github.com/moment-timeseries-foundation-model/moment) |
| 2024 | **Chronos: Learning the Language of Time Series** | ICML | 将时序视为语言，基于Transformer的预测模型 | [PDF](https://arxiv.org/pdf/2403.07815) \| [Code](https://github.com/amazon-science/chronos-forecasting) |
| 2025 | **Lag-Llama: Towards Foundation Models for Probabilistic...** | ICLR | 概率时间序列预测的基础模型 | [PDF](https://arxiv.org/pdf/2310.08278) \| [Code](https://github.com/time-series-foundation-models/lag-llama) |
| 2024 | UniTS: Building a Unified Time Series Model | NeurIPS | 统一多任务时间序列模型 | [PDF](https://arxiv.org/pdf/2403.00131) \| [Code](https://github.com/mims-harvard/UniTS) |

## 自回归与跨域迁移 (Next-Gen AR & Cross-Domain)
> 聚焦 AR 范式的五大方向：尺度生成、离散/潜在空间、连续空间、测试时记忆与函数式叙事。
> **Domain**: CV/Video, NLP/General, Time Series (Target)

|  年份 |  核心机制 (Mechanism) | 🌌领域 |  标题 (Title) | 💡推荐理由/迁移点 |  资源 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **2026** | **Scale-wise (尺度递进)** |  TS | **TimeMAR: Multi-Scale Autoregressive Modeling** | **[必读]** 迁移自 VAR，从粗糙(Trend)到精细(Seasonality)的生成 | [PDF](https://arxiv.org/pdf/2601.11184) |
| 2024 | Scale-wise (尺度递进) |  CV | VAR: Visual Autoregressive Modeling | **[原型]** 下一尺度预测开山之作，速度比 Diffusion 快 20 倍 | [PDF](https://arxiv.org/pdf/2404.02905) |
| **2024** | **Discrete VQ (离散化)** |  TS | **SDformer: Similarity-driven Discrete Transformer** | **[SOTA]** 提出相似度驱动的 VQ 机制，推理速度比 Diffusion 快 30 倍 | [PDF]_(https://proceedings.mlr.press/v235/gao24c/gao24c.pdf) \| [Code](https://github.com/Spongebob-Hui/SDformer) |
| **2025** | **Multi-scale VQ (多尺度离散)** |  TS | **MSDformer: Multi-scale Discrete Transformer** | **[进阶]** SDformer 的升级版，引入级联残差 VQ-VAE 捕捉多尺度特征 | [PDF](https://arxiv.org/pdf/2505.14202) |
| **2025** | **Latent AR (潜在空间)** |  TS | **Less Is More: Generating Time Series with LLaMA...** | **[新范式]** 结合 LLaMA 架构与简化因子化潜在空间，追求极致效率 | [PDF](链接) |
| **2025** | **Functional (函数叙事)** |  TS | **NoTS: Generalizable AR Modeling Through Functional Narratives** | **[必读]** Apple出品，将时序视为函数序列，引入退化算子 | [PDF_(https://arxiv.org/pdf/2410.08421)|
| 2025 | **Continuous (非量化)** |  Video | NOVA: AR Video Generation without Vector Quantization | **[原型]** 抛弃 VQ-VAE，解决量化带来的高频数值精度丢失问题 | [PDF](https://arxiv.org/pdf/2401.12945) |
| 2025 | **Test-Time Memory** |  AI | Titans: Learning to Memorize at Test Time | **[原型]** Google新架构，测试时实时更新记忆，对抗 Concept Drift | [PDF](https://arxiv.org/pdf/2501.00663) |
| 2025 | **Masked AR (掩码)** |  CV | HMAR: Efficient Hierarchical Masked Auto-Regressive | **[原型]** 结合双向能力，适合做任意条件(Any-condition)的时序补全 | [Link](https://arxiv.org/abs/2403.13731) |

##  基于扩散模型的生成 (Diffusion-based Generation)
> 利用 Diffusion/Score-based Model 进行高质量、可解释的时间序列生成。

|  年份 | 标题 (Title) |  会议/期刊 |  核心点/备注 |  资源 |
| :--- | :--- | :--- | :--- | :--- |
| **2026** | **DLTTS: Diffusion Model for Long-Tailed...** | IEEE TKDE | **[工业/长尾]** 结合 Informer 与 FBMC Loss，专攻工业长尾数据生成 | [Paper](https://ieeexplore.ieee.org/abstract/document/10834079) |
| **2025** | **MA-TSD: A Non-isotropic Time Series Diffusion Model...** | ICML | **[SOTA]** 利用移动平均(MA)作为前向过程，保留低频信息，解决梯度冲突 | [PDF](https://openreview.net/pdf?id=yFF7hy8Mbh) |
| **2024** | **ARMD: Auto-Regressive Moving Diffusion Models...** | IJCAI | **[混合架构]** 将自回归移动过程融入扩散去噪，增强时间依赖性捕捉 | [PDF](https://www.ijcai.org/proceedings/2024/0523.pdf) \| [Code](https://github.com/zhycs/ARMD) |
| **2025** | **ImagenI2R: A Diffusion Model for Regular TS Generation...** | NeurIPS | **[SOTA]** 两步法：先用Transformer补全，再用视觉Diffusion生成，专攻非规则数据 | [PDF](https://arxiv.org/pdf/2510.06699) \| [Code](https://github.com/azencot-group/ImagenI2R) |
| 2025 | **SigDiffusions: Score-Based Diffusion Models...** | ICLR | **[理论流派]** 在 Log-Signature 空间扩散，利用粗糙路径理论捕捉长程依赖 | [PDF](https://arxiv.org/pdf/2406.10354) \| [Code](https://github.com/Barb0ra/SigDiffusions) |
| 2025 | **TSGM: Universal Time-series Generation using Score-based...** | ICLR | 基于分数的通用生成模型，支持非规则序列 | [PDF](https://arxiv.org/pdf/2511.21335) |
| 2024 | **Diffusion-TS: Interpretable Diffusion for General Time Series** | ICLR | 提供可解释性的通用扩散生成框架 | [PDF](https://arxiv.org/pdf/2403.01742) \| [Code](https://github.com/Y-debug-sys/Diffusion-TS) |
| 2024 | FIDE: Frequency-Inflated Conditional Diffusion Model... | NeurIPS | 频率增强条件扩散，针对极端感知生成 | [PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/cfce727868dcaf5295c0125f9d6fbc0b-Paper-Conference.pdf) \| [Code](https://github.com/galib19/FIDE) |
| 2025 | Ctrl-Adapter: An Efficient and Versatile Framework... | ICLR | 适配任意扩散模型的高效控制框架 | [PDF](https://arxiv.org/pdf/2404.09967) \| [Code](https://github.com/HL-hanlin/Ctrl-Adapter) |
| 2025 | Diffusion Transformers for Tabular Data Time Series Generation | ICLR | 表格数据时间序列生成的 DiT 应用 | [PDF](https://arxiv.org/pdf/2504.07566) \| [Code](https://github.com/fabriziogaruti/TabDiT) |
| 2025 | Population Aware Diffusion for Time Series Generation | AAAI | 群体统计特征感知的生成 | [PDF](https://arxiv.org/pdf/2501.00910) \| [Code](https://github.com/wmd3i/PaD-TS) |
| 2024 | TimeLDM: Latent Diffusion Model for Unconditional... | Preprint | 效率优化与长序列生成（潜在扩散模型） | [PDF](https://arxiv.org/pdf/2407.04211) |

##  文本驱动与可控生成 (Text-to-Series & Controllable)
> 通过文本描述或特定指令来生成/编辑时间序列。

|  年份 |  标题  | 会议 |  核心点/备注 |  资源 |
| :--- | :--- | :--- | :--- | :--- |
| 2025 | **VerbaTS: Generating Time Series from Texts** | ICML | 文本交互生成时间序列 (VerbalTS) | [PDF](https://proceedings.mlr.press/v267/gu25a/gu25a.pdf) \| [Code](https://github.com/seqml/VerbaTS) |
| 2025 | **T2S: High-resolution Time Series Generation with Text...** | IJCAI | 文本到序列扩散模型，高分辨率生成 | [PDF](https://arxiv.org/pdf/2505.02417) \| [Code](https://github.com/WinfredGc/T2S) |
| 2025 | TimeDP: Learning to Generate Multi-Domain Time Series... | AAAI | 利用领域提示词生成多域时序 | [PDF](https://arxiv.org/pdf/2501.05403) \| [Code](https://github.com/microsoft/TimeCraft) |
| 2024 | Towards Editing Time Series | NeurIPS | 时间序列编辑（局部修改趋势） | [PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/423d0909791493b7c10916fd328c2913-Paper-Conference.pdf) \| [Code](https://github.com/seqml/TEdit) |

##  通用生成与分解 (General Synthesis & Decomposition)
> 关注数据分解、稀缺性问题及通用生成技术。

|  年份 | 标题  |  会议 |  核心点/备注 |  资源 |
| :--- | :--- | :--- | :--- | :--- |
| 2025 | Effective Series Decomposition and Components Learning... | ICDM | 周期性、趋势性扩散模型 (STDiffusion) | [PDF](https://arxiv.org/pdf/2511.00747) |
| 2024 | Generative Time Series Forecasting with Diffusion... | NeurIPS | 结合分解、去噪、解缠的生成式预测 | [PDF](https://arxiv.org/pdf/2301.03028) \| [Code](https://github.com/PaddlePaddle/PaddleSpatial) |
| 2025 | Time Series Generation Under Data Scarcity... | NeurIPS | 数据稀缺下的统一生成建模方法 | [PDF](https://arxiv.org/pdf/2505.20446) \| [Code](https://github.com/azencot-group/ImagenFew) |

---
*Generated based on user research list.*
