# Awesome Vision Large Language Models - Methods and Applications

A collection of papers on Vision Large Language Models and their Applications.

## Table of Contents
- [Awesome Vision Large Language Models - Methods and Applications](#awesome-vision-large-language-models---methods-and-applications)
  - [Table of Contents](#table-of-contents)
  - [Surveys and Evaluations](#surveys-and-evaluations)
  - [Architectural Innovations](#architectural-innovations)
    - [Multimodal Foundation Models](#multimodal-foundation-models)
    - [Projection and Fusion Mechanisms](#projection-and-fusion-mechanisms)
    - [Scaling Strategies](#scaling-strategies)
    - [Perception Control](#perception-control)
    - [Model Compression](#model-compression)
    - [Representation Learning](#representation-learning)
    - [Reasoning and Sequential Processing](#reasoning-and-sequential-processing)
    - [Data Scaling and Curation](#data-scaling-and-curation)
    - [Self-Supervised Learning](#self-supervised-learning)
  - [Training and Alignment](#training-and-alignment)
    - [Pre-training Objectives](#pre-training-objectives)
    - [Instruction Tuning](#instruction-tuning)
    - [Reinforcement Learning Alignment](#reinforcement-learning-alignment)
  - [Advanced Reasoning Capabilities](#advanced-reasoning-capabilities)
    - [Visual Chain-of-Thought](#visual-chain-of-thought)
    - [In-Context Learning](#in-context-learning)
    - [Hallucination Mitigation](#hallucination-mitigation)
    - [Spatial Reasoning](#spatial-reasoning)
  - [Application Domains](#application-domains)
    - [Video Understanding](#video-understanding)
    - [Embodied Intelligence](#embodied-intelligence)
    - [Autonomous Driving](#autonomous-driving)
    - [Document and Chart Understanding](#document-and-chart-understanding)
    - [Healthcare](#healthcare)
    - [Smart Cities](#smart-cities)
    - [Gaming AI](#gaming-ai)
    - [GUI Agents](#gui-agents)
  - [Evaluation and Benchmarks](#evaluation-and-benchmarks)
  - [Security and Robustness](#security-and-robustness)
  - [Future Challenges](#future-challenges)

## Surveys and Evaluations

| Title | Focus | Date | Link |
|-------|-------|------|------|
| [**Large Vision-Language Model Alignment and Misalignment: A Survey Through the Lens of Explainability**](https://arxiv.org/abs/2501.01346) | Alignment & Explainability | 2024-01 | [Paper](https://arxiv.org/abs/2501.01346) |
| [**Visual Large Language Models for Generalized and Specialized Applications**](https://arxiv.org/abs/2501.02765) | Applications & Ethics | 2024-01 | [Paper](https://arxiv.org/abs/2501.02765) |
| [**A Survey on Backdoor Threats in Large Language Models (LLMs): Attacks, Defenses, and Evaluations**](https://arxiv.org/abs/2502.05224) | Security & Robustness | 2024-02 | [Paper](https://arxiv.org/abs/2502.05224) |
| [**A Survey of State of the Art Large Vision Language Models: Alignment, Benchmark, Evaluations and Challenges**](https://arxiv.org/abs/2501.02189) | Benchmarks & Challenges | 2024-01 | [Paper](https://arxiv.org/abs/2501.02189) |
| [**Vision-Language Models for Edge Networks: A Comprehensive Survey**](https://arxiv.org/abs/2502.07855) | Deployment & Edge AI | 2024-02 | [Paper](https://arxiv.org/abs/2502.07855) |
| [**Vision-Language Model for Object Detection and Segmentation: A Review and Evaluation**](https://arxiv.org/abs/2504.09480) | Detection & Segmentation | 2024-04 | [Paper](https://arxiv.org/abs/2504.09480) |
| [**A Survey on Efficient Vision-Language Models**](https://arxiv.org/abs/2504.09724) | Efficiency & Edge AI | 2024-04 | [Paper](https://arxiv.org/abs/2504.09724) |

## Architectural Innovations

### Multimodal Foundation Models

| Title | Venue | Date | Code | Project |
|-------|-------|------|------|---------|
| [**InternVL3: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models**](https://arxiv.org/abs/2504.10479) | arXiv | 2025-04-14 | [Github](https://github.com/OpenGVLab/InternVL) | [Demo](https://internvl.opengvlab.com/) |
| [**The Llama 4 herd: The beginning of a new era of natively multimodal AI innovation**](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) | Meta | 2025-04-05 | [Hugging Face](https://huggingface.co/collections/meta-llama/llama-4-67f0c30d9fe03840bc9d0164) | - |
| [**Qwen2.5-Omni Technical Report**](https://github.com/QwenLM/Qwen2.5-Omni/blob/main/assets/Qwen2.5_Omni.pdf) | Qwen | 2025-03-26 | [Github](https://github.com/QwenLM/Qwen2.5-Omni) | [Demo](https://huggingface.co/spaces/Qwen/Qwen2.5-Omni-7B-Demo) |
| [**Qwen2.5-VL Technical Report**](https://arxiv.org/pdf/2502.13923) | arXiv | 2025-02-19 | [Github](https://github.com/QwenLM/Qwen2.5-VL) | [Demo](https://huggingface.co/spaces/Qwen/Qwen2.5-VL) |
| [**Gemini: A Family of Highly Capable Multimodal Models**](https://storage.googleapis.com/deepmind-media/gemini/gemini_1_report.pdf) | Google | 2023-12-06 | - | - |
| [**GPT-4V**](https://arxiv.org/pdf/2309.17421) | - | 2023 | - | - |
| [**LLaVA: Visual Instruction Tuning**](https://arxiv.org/abs/2304.08485) | NeurIPS | 2023-04-17 | [Github](https://github.com/haotian-liu/LLaVA) | [Project](https://llava-vl.github.io/) |
| [**BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models**](https://arxiv.org/pdf/2301.12597) | ICML | 2023 | [Github](https://github.com/salesforce/LAVIS/tree/main/projects/blip2) | - |
| [**Flamingo: a Visual Language Model for Few-Shot Learning**](https://arxiv.org/pdf/2204.14198) | NeurIPS | 2022-04-29 | [Github](https://github.com/mlfoundations/open_flamingo) | - |
| [**CLIP: Learning Transferable Visual Models From Natural Language Supervision**](https://arxiv.org/abs/2103.00020) | ICML | 2021 | [Github](https://github.com/OpenAI/CLIP) | [Project](https://openai.com/index/clip/) |

### Projection and Fusion Mechanisms

| Title | Venue | Date | Code | Project |
|-------|-------|------|------|---------|
| [**ConvLLaVA: Hierarchical Backbones as Visual Encoder for Large Multimodal Models**](https://arxiv.org/pdf/2405.15738) | arXiv | 2024-05-24 | [Github](https://github.com/alibaba/conv-llava) | - |
| [**CuMo: Scaling Multimodal LLM with Co-Upcycled Mixture-of-Experts**](https://arxiv.org/pdf/2405.05949) | arXiv | 2024-05-09 | [Github](https://github.com/SHI-Labs/CuMo) | - |
| [**MoE-LLaVA: Mixture of Experts for Large Vision-Language Models**](https://arxiv.org/pdf/2401.15947) | arXiv | 2024-01-29 | [Github](https://github.com/PKU-YuanGroup/MoE-LLaVA) | [Demo](https://huggingface.co/spaces/LanguageBind/MoE-LLaVA) |
| [**VILA: On Pre-training for Visual Language Models**](https://arxiv.org/pdf/2312.07533) | CVPR | 2023-12-13 | [Github](https://github.com/NVlabs/VILA) | - |
| [**Honeybee: Locality-enhanced Projector for Multimodal LLM**](https://arxiv.org/pdf/2312.06742) | CVPR | 2023-12-11 | [Github](https://github.com/kakaobrain/honeybee) | - |
| [**LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention**](https://arxiv.org/abs/2303.16199) | ICLR | 2023-03-28 | [Github](https://github.com/ZrrSkywalker/LLaMA-Adapter) | - |

### Scaling Strategies

| Title | Venue | Date | Code | Project |
|-------|-------|------|------|---------|
| [**Long-VITA: Scaling Large Multi-modal Models to 1 Million Tokens with Leading Short-Context Accuray**](https://arxiv.org/pdf/2502.05177) | arXiv | 2025-02-19 | [Github](https://github.com/VITA-MLLM/Long-VITA) | - |
| [**Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling**](https://arxiv.org/pdf/2412.05271) | arXiv | 2024-12-06 | [Github](https://github.com/OpenGVLab/InternVL) | [Demo](https://internvl.opengvlab.com) |
| [**EAGLE: Exploring The Design Space for Multimodal LLMs with Mixture of Encoders**](https://arxiv.org/pdf/2408.15998) | arXiv | 2024-08-28 | [Github](https://github.com/NVlabs/Eagle) | [Demo](https://huggingface.co/spaces/NVEagle/Eagle-X5-13B-Chat) |
| [**LongLLaVA: Scaling Multi-modal LLMs to 1000 Images Efficiently via Hybrid Architecture**](https://arxiv.org/pdf/2409.02889) | arXiv | 2024-09-04 | [Github](https://github.com/FreedomIntelligence/LongLLaVA) | - |

### Perception Control

| Title | Venue | Date | Code | Project |
|-------|-------|------|------|---------|
| [**Introducing Visual Perception Token into Multimodal Large Language Model**](https://arxiv.org/abs/2502.17425) | arXiv | 2024-02 | - | - |
| [**Your Large Vision-Language Model Only Needs A Few Attention Heads For Visual Grounding**](https://arxiv.org/abs/2503.06287) | arXiv | 2024-03 | - | - |
| [**LongProLIP: A Probabilistic Vision-Language Model with Long Context Text**](https://arxiv.org/abs/2503.08048) | arXiv | 2024-03 | - | - |
| [**Vision-R1: Incentivizing Reasoning Capability in Multimodal Large Language Models**](https://arxiv.org/abs/2503.06749) | arXiv | 2024-03 | - | - |
| [**The Scalability of Simplicity: Empirical Analysis of Vision-Language Learning with a Single Transformer**](https://arxiv.org/abs/2504.10462) | arXiv | 2024-04 | - | - |

### Model Compression

| Title | Venue | Date | Code | Project |
|-------|-------|------|------|---------|
| [**EfficientLLaVA: Generalizable Auto-Pruning for Large Vision-language Models**](https://arxiv.org/abs/2503.15369) | arXiv | 2024-03 | - | - |
| [**Scalable Vision Language Model Training via High Quality Data Curation**](https://arxiv.org/abs/2501.05952) | arXiv | 2024-01 | - | - |
| [**Granite Vision: a lightweight, open-source multimodal model for enterprise Intelligence**](https://arxiv.org/abs/2502.09927) | arXiv | 2024-02 | [GitHub](https://github.com/ibm-granite/granite-vision-models) | - |
| [**NanoVLMs: How Small Can We Go and Still Make Coherent Vision Language Models?**](https://arxiv.org/abs/2502.07838) | arXiv | 2024-02 | [GitHub](https://github.com/eisneim/nanoVLM) | - |

### Representation Learning

| Title | Venue | Date | Code | Project |
|-------|-------|------|------|---------|
| [**Multi-Modal Representation Learning for Vision-Language Models**](https://arxiv.org/abs/2503.08497) | arXiv | 2024-03 | - | - |
| [**Unifying 2D and 3D Vision-Language Understanding**](https://arxiv.org/abs/2503.10745) | arXiv | 2024-03 | - | - |

### Reasoning and Sequential Processing

| Title | Venue | Date | Code | Project |
|-------|-------|------|------|---------|
| [**GFlowVLM: Enhancing Multi-Step Reasoning in Vision-Language Models**](https://arxiv.org/abs/2503.06514) | arXiv | 2024-03 | - | - |
| [**SDRT: Enhance Vision-Language Models by Self-Distillation with Diverse Reasoning Traces**](https://arxiv.org/abs/2503.01754) | arXiv | 2024-03 | - | - |
| [**Rethinking RL Scaling for Vision-Language Models**](https://arxiv.org/abs/2504.02587) | arXiv | 2024-04 | - | - |
| [**A Vision-Language-Action Model with Open-World Generalization**](https://arxiv.org/abs/2504.16054) | arXiv | 2024-04 | - | - |

### Data Scaling and Curation

| Title | Venue | Date | Code | Project |
|-------|-------|------|------|---------|
| [**Scaling Pre-training to One Hundred Billion Data for Vision-Language Models**](https://arxiv.org/abs/2502.07617) | arXiv | 2024-02 | - | - |

### Self-Supervised Learning

| Title | Venue | Date | Code | Project |
|-------|-------|------|------|---------|
| [**Vision-Language Model Dialog Games for Self-Improvement**](https://arxiv.org/abs/2502.02740) | arXiv | 2024-02 | - | - |

## Training and Alignment

### Pre-training Objectives

| Title | Venue | Date | Code | Project |
|-------|-------|------|------|---------|
| [**ALIP: Adaptive Language-Image Pre-training with Synthetic Caption**](https://arxiv.org/pdf/2308.08428.pdf) | ICCV | 2023 | [Github](https://github.com/deepglint/ALIP) | - |
| [**RA-CLIP: Retrieval Augmented Contrastive Language-Image Pre-training**](https://openaccess.thecvf.com/content/CVPR2023/papers/Xie_RA-CLIP_Retrieval_Augmented_Contrastive_Language-Image_Pre-Training_CVPR_2023_paper.pdf) | CVPR | 2023 | - | - |
| [**GroupViT: Semantic Segmentation Emerges from Text Supervision**](https://arxiv.org/abs/2202.11094) | CVPR | 2022 | [Github](https://github.com/NVlabs/GroupViT) | - |
| [**LiT: Zero-Shot Transfer with Locked-image text Tuning**](https://arxiv.org/abs/2111.07991) | CVPR | 2022 | [Code](https://google-research.github.io/vision_transformer/lit/) | - |
| [**ALIGN: Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision**](https://arxiv.org/pdf/2102.05918.pdf) | ICML | 2021 | - | - |

### Instruction Tuning

| Title | Venue | Date | Code | Project |
|-------|-------|------|------|---------|
| [**Inst-IT: Boosting Multimodal Instance Understanding via Explicit Visual Prompt Instruction Tuning**](https://arxiv.org/abs/2412.03565) | arXiv | 2024-12-04 | [Github](https://github.com/inst-it/inst-it) | - |
| [**OMG-LLaVA: Bridging Image-level, Object-level, Pixel-level Reasoning and Understanding**](https://arxiv.org/pdf/2406.19389) | arXiv | 2024-06-27 | [Github](https://github.com/lxtGH/OMG-Seg) | - |
| [**LLaVA-1.5: Improved Baselines with Visual Instruction Tuning**](https://arxiv.org/abs/2310.03744) | CVPR | 2024 | [Github](https://github.com/haotian-liu/LLaVA) | [Project](https://llava-vl.github.io/) |
| [**InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning**](https://arxiv.org/pdf/2305.06500) | arXiv | 2023-05-11 | [Github](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip) | - |
| [**SVIT: Scaling up Visual Instruction Tuning**](https://arxiv.org/pdf/2307.04087) | arXiv | 2023-07-09 | [Github](https://github.com/BAAI-DCAI/Visual-Instruction-Tuning) | - |

### Reinforcement Learning Alignment

| Title | Venue | Date | Code | Project |
|-------|-------|------|------|---------|
| [**VideoChat-R1: Enhancing Spatio-Temporal Perception via Reinforcement Fine-Tuning**](https://arxiv.org/abs/2504.06958) | arXiv | 2025-04-10 | [Github](https://github.com/OpenGVLab/VideoChat-R1) | - |
| [**OpenVLThinker: An Early Exploration to Complex Vision-Language Reasoning via Iterative Self-Improvement**](https://arxiv.org/abs/2503.17352) | arXiv | 2025-03-21 | [Github](https://github.com/yihedeng9/OpenVLThinker) | - |
| [**Boosting the Generalization and Reasoning of Vision Language Models with Curriculum Reinforcement Learning**](https://arxiv.org/abs/2503.07065) | arXiv | 2025-03-10 | [Github](https://github.com/ding523/Curr_REFT) | - |
| [**OmniAlign-V: Towards Enhanced Alignment of MLLMs with Human Preference**](https://arxiv.org/abs/2502.18411) | arXiv | 2025 | [Github](https://github.com/PhoenixZ810/OmniAlign-V) | - |
| [**MM-RLHF: The Next Step Forward in Multimodal LLM Alignment**](https://arxiv.org/pdf/2502.10391) | arXiv | 2025 | [Github](https://github.com/Kwai-YuanQi/MM-RLHF) | - |
| [**Aligning Large Multimodal Models with Factually Augmented RLHF**](https://arxiv.org/abs/2309.14525) | arXiv | 2023-09-25 | [Github](https://github.com/llava-rlhf/LLaVA-RLHF) | [Project](https://llava-rlhf.github.io/) |

## Advanced Reasoning Capabilities

### Visual Chain-of-Thought

| Title | Venue | Date | Code | Project |
|-------|-------|------|------|---------|
| [**Insight-V: Exploring Long-Chain Visual Reasoning with Multimodal Large Language Models**](https://arxiv.org/pdf/2411.14432) | arXiv | 2024-11-21 | [Github](https://github.com/dongyh20/Insight-V) | - |
| [**Cantor: Inspiring Multimodal Chain-of-Thought of MLLM**](https://arxiv.org/pdf/2404.16033.pdf) | arXiv | 2024-04-24 | [Github](https://github.com/ggg0919/cantor) | - |
| [**Visual CoT: Unleashing Chain-of-Thought Reasoning in Multi-Modal Language Models**](https://arxiv.org/pdf/2403.16999.pdf) | arXiv | 2024-03-25 | [Github](https://github.com/deepcs233/Visual-CoT) | - |
| [**Compositional Chain-of-Thought Prompting for Large Multimodal Models**](https://arxiv.org/pdf/2311.17076) | CVPR | 2023-11-27 | [Github](https://github.com/chancharikmitra/CCoT) | - |
| [**Multimodal Chain-of-Thought Reasoning in Language Models**](https://arxiv.org/pdf/2302.00923.pdf) | arXiv | 2023-02-02 | [Github](https://github.com/amazon-science/mm-cot) | - |
| [**Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering**](https://proceedings.neurips.cc/paper_files/paper/2022/file/11332b6b6cf4485b84afadb1352d3a9a-Paper-Conference.pdf) | NeurIPS | 2022-09-20 | [Github](https://github.com/lupantech/ScienceQA) | - |

### In-Context Learning

| Title | Venue | Date | Code | Project |
|-------|-------|------|------|---------|
| [**ICL-D3IE: In-Context Learning with Diverse Demonstrations Updating for Document Information Extraction**](https://arxiv.org/pdf/2303.05063.pdf) | ICCV | 2023-03-09 | [Github](https://github.com/MAEHCM/ICL-D3IE) | - |
| [**Prompting Large Language Models with Answer Heuristics for Knowledge-based Visual Question Answering**](https://arxiv.org/pdf/2303.01903.pdf) | CVPR | 2023-03-03 | [Github](https://github.com/MILVLG/prophet) | - |
| [**Visual Programming: Compositional visual reasoning without training**](https://openaccess.thecvf.com/content/CVPR2023/papers/Gupta_Visual_Programming_Compositional_Visual_Reasoning_Without_Training_CVPR_2023_paper.pdf) | CVPR | 2022-11-18 | [Github](https://github.com/allenai/visprog) | - |
| [**An Empirical Study of GPT-3 for Few-Shot Knowledge-Based VQA**](https://ojs.aaai.org/index.php/AAAI/article/download/20215/19974) | AAAI | 2022-06-28 | [Github](https://github.com/microsoft/PICa) | - |
| [**Multimodal Few-Shot Learning with Frozen Language Models**](https://arxiv.org/pdf/2106.13884.pdf) | NeurIPS | 2021-06-25 | - | - |
| [**Large (Vision) Language Models are Unsupervised In-Context Learners**](https://arxiv.org/abs/2504.02349) | arXiv | 2024-04 | - | - |

### Hallucination Mitigation

| Title | Venue | Date | Code | Project |
|-------|-------|------|------|---------|
| [**Woodpecker: Hallucination Correction for Multimodal Large Language Models**](https://arxiv.org/pdf/2310.16045) | arXiv | 2023 | [Github](https://github.com/BradyFU/Woodpecker) | - |
| [**POPE: Evaluating Object Hallucination in Large Vision-Language Models**](https://arxiv.org/pdf/2305.10355) | arXiv | 2023 | [Github](https://github.com/RUCAIBox/POPE) | - |
| [**HallusionBench: Benchmarking Hallucinations for Multimodal LLMs**](https://arxiv.org/pdf/2310.14566) | arXiv | 2023 | [Github](https://github.com/tianyi-lab/HallusionBench) | - |
| [**HallE-Control: Hallucination Detection and Control for Large Vision-Language Models**](https://arxiv.org/abs/2310.01779) | arXiv | 2023 | [Github](https://github.com/bronyayang/HallE_Control) | - |

### Spatial Reasoning

| Title | Venue | Date | Code | Project |
|-------|-------|------|------|---------|
| [**Vision language models are unreliable at trivial spatial cognition**](https://arxiv.org/abs/2504.16061) | arXiv | 2024-04 | - | - |
| [**Why Vision Language Models Struggle with Visual Arithmetic? Towards Enhanced Chart and Geometry Understanding**](https://arxiv.org/abs/2502.11492) | arXiv | 2024-02 | - | - |
| [**Do Vision-Language Models Have Blind Faith in Text?**](https://arxiv.org/abs/2503.02199) | arXiv | 2024-03 | [GitHub](https://github.com/d-ailin/blind-faith-in-text) | - |

## Application Domains

### Video Understanding

| Title | Venue | Date | Code | Project |
|-------|-------|------|------|---------|
| [**Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis**](https://arxiv.org/pdf/2405.21075) | arXiv | 2024 | [Github](https://github.com/BradyFU/Video-MME) | [Project](https://video-mme.github.io/) |
| [**TimeMarker: A Versatile Video-LLM for Long and Short Video Understanding with Superior Temporal Localization Ability**](https://arxiv.org/pdf/2411.18211) | arXiv | 2024-11-27 | [Github](https://github.com/TimeMarker-LLM/TimeMarker/) | - |
| [**LongVU: Spatiotemporal Adaptive Compression for Long Video-Language Understanding**](https://arxiv.org/pdf/2410.17434) | arXiv | 2024-10-22 | [Github](https://github.com/Vision-CAIR/LongVU) | [Demo](https://huggingface.co/spaces/Vision-CAIR/LongVU) |
| [**VideoLLM-online: Online Video Large Language Model for Streaming Video**](https://arxiv.org/pdf/2406.11816) | CVPR | 2024-06-17 | [Github](https://github.com/showlab/VideoLLM-online) | - |
| [**VideoLLaMA 2: Advancing Spatial-Temporal Modeling and Audio Understanding in Video-LLMs**](https://arxiv.org/pdf/2406.07476) | arXiv | 2024-06-11 | [Github](https://github.com/DAMO-NLP-SG/VideoLLaMA2) | - |
| [**Video-LLaVA: Learning United Visual Representation by Alignment Before Projection**](https://arxiv.org/pdf/2311.10122) | arXiv | 2023-11-16 | [Github](https://github.com/PKU-YuanGroup/Video-LLaVA) | [Demo](https://huggingface.co/spaces/LanguageBind/Video-LLaVA) |
| [**VideoLLaMA 3: Frontier Multimodal Foundation Models for Image and Video Understanding**](https://arxiv.org/abs/2501.13106) | arXiv | 2024-01 | [GitHub](https://github.com/DAMO-NLP-SG/VideoLLaMA3) | - |

### Embodied Intelligence

| Title | Venue | Date | Code | Project |
|-------|-------|------|------|---------|
| [**RoboPoint: A Vision-Language Model for Spatial Affordance Prediction for Robotics**](https://arxiv.org/pdf/2406.10721) | CoRL | 2024-06-15 | [Github](https://github.com/wentaoyuan/RoboPoint) | [Demo](https://007e03d34429a2517b.gradio.live/) |
| [**An Embodied Generalist Agent in 3D World**](https://arxiv.org/pdf/2311.12871.pdf) | arXiv | 2023-11-18 | [Github](https://github.com/embodied-generalist/embodied-generalist) | [Demo](https://www.youtube.com/watch?v=mlnjz4eSjB4) |
| [**EmbodiedGPT: Vision-Language Pre-Training via Embodied Chain of Thought**](https://arxiv.org/pdf/2305.15021.pdf) | arXiv | 2023-05-24 | [Github](https://github.com/EmbodiedGPT/EmbodiedGPT_Pytorch) | - |
| [**PaLM-E: Embodied Multimodal Language Models**](https://arxiv.org/pdf/2303.03378) | arXiv | 2023 | - | - |

### Autonomous Driving

| Title | Venue | Date | Code | Project |
|-------|-------|------|------|---------|
| [**Senna: Bridging Large Vision-Language Models and End-to-End Autonomous Driving**](https://arxiv.org/pdf/2410.22313) | arXiv | 2024-10-29 | [Github](https://github.com/hustvl/Senna) | - |
| [**DriveLM: Driving with Graph Visual Question Answering**](https://arxiv.org/pdf/2312.14150) | ECCV | 2024-7-17 | [Github](https://github.com/OpenDriveLab/DriveLM) | - |
| [**Dolphins: Multimodal Language Model for Driving**](https://arxiv.org/pdf/2312.00438.pdf) | arXiv | 2023-12-01 | [Github](https://github.com/vlm-driver/Dolphins) | - |
| [**DriveGPT4: Interpretable End-to-End Autonomous Driving Via Large Language Model**](https://arxiv.org/pdf/2311.13549) | RAL | 2024-8-7 | [Github](https://drive.google.com/drive/folders/1PsGL7ZxMMz1ZPDS5dZSjzjfPjuPHxVL5?usp=sharing) | [Project](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10629039) |
| [**DriveVLM: The Convergence of Autonomous Driving and Large Vision-Language Models**](https://arxiv.org/abs/2402.12289) | CoRL | 2024-6-25 | - | [Project](https://tsinghua-mars-lab.github.io/DriveVLM/) |

### Document and Chart Understanding

| Title | Venue | Date | Code | Project |
|-------|-------|------|------|---------|
| [**DocKylin: A Large Multimodal Model for Visual Document Understanding with Efficient Visual Slimming**](https://arxiv.org/pdf/2406.19101) | AAAI | 2024-06-27 | [Github](https://github.com/ZZZHANG-jx/DocKylin) | - |
| [**Beyond Embeddings: The Promise of Visual Table in Multi-Modal Models**](https://arxiv.org/pdf/2403.18252.pdf) | arXiv | 2024-03-27 | [Github](https://github.com/LaVi-Lab/Visual-Table) | - |
| [**TextMonkey: An OCR-Free Large Multimodal Model for Understanding Document**](https://arxiv.org/pdf/2403.04473.pdf) | arXiv | 2024-03-07 | [Github](https://github.com/Yuliang-Liu/Monkey) | [Demo](http://vlrlab-monkey.xyz:7684) |
| [**ChartLlama: A Multimodal LLM for Chart Understanding and Generation**](https://arxiv.org/pdf/2311.16483.pdf) | arXiv | 2023-11-27 | [Github](https://github.com/tingxueronghua/ChartLlama-code) | - |
| [**ChartAssisstant: A Universal Chart Multimodal Language Model via Chart-to-Table Pre-training and Multitask Instruction Tuning**](https://arxiv.org/pdf/2401.02384) | ACL | 2024-01-04 | [Github](https://github.com/OpenGVLab/ChartAst) | - |

### Healthcare

| Title | Venue | Date | Code | Project |
|-------|-------|------|------|---------|
| [**Multi-Modal One-Shot Federated Ensemble Learning for Medical Data with Vision Large Language Model**](https://arxiv.org/abs/2501.03292) | arXiv | 2024-01 | - | - |
| [**Evaluating Vision-Language Models (VLMs) for Radiology**](https://arxiv.org/abs/2504.16047) | arXiv | 2024-04 | - | - |

### Smart Cities

| Title | Venue | Date | Code | Project |
|-------|-------|------|------|---------|
| [**Urban Road Anomaly Monitoring Using Vision–Language Models for Enhanced Safety Management**](https://doi.org/10.3390/app15052517) | Applied Sciences | 2024 | - | - |

### Gaming AI

| Title | Venue | Date | Code | Project |
|-------|-------|------|------|---------|
| [**Are Large Vision Language Models Good Game Players?**](https://arxiv.org/abs/2503.02358) | arXiv | 2024-03 | - | - |

### GUI Agents

| Title | Venue | Date | Code | Project |
|-------|-------|------|------|---------|
| [**GUI-R1: A Generalist R1-Style Vision-Language Action Model For GUI Agents**](https://arxiv.org/abs/2504.10458) | arXiv | 2024-04 | - | - |

## Evaluation and Benchmarks

| Title | Domain | Date | Project |
|-------|--------|------|---------|
| [**Video SimpleQA**](https://arxiv.org/abs/2503.18923) | Video understanding | 2024 | [Github](https://videosimpleqa.github.io) |
| [**VisionArena**](https://arxiv.org/abs/2412.08687) | Multimodal conversation | 2024 | [Hugging Face](https://huggingface.co/lmarena-ai) |
| [**Inst-IT-Bench**](https://arxiv.org/abs/2412.03565) | Fine-grained Image and Video Understanding | 2024 | [Github](https://github.com/inst-it/inst-it) |
| [**MMMU**](https://arxiv.org/pdf/2311.16502) | Multimodal reasoning and understanding | 2023 | [Website](https://mmmu-benchmark.github.io/) |
| [**MM-Vet**](https://arxiv.org/pdf/2308.02490) | OCR, visual reasoning | 2023 | [Github](https://github.com/yuweihao/MM-Vet) |
| [**MMBench**](https://arxiv.org/abs/2307.06281) | Multilingual multimodal understanding | 2023 | [Github](https://github.com/open-compass/MMBench) |
| [**HallusionBench**](https://arxiv.org/pdf/2310.14566) | Hallucination | 2023 | [Github](https://github.com/tianyi-lab/HallusionBench) |
| [**Where do Large Vision-Language Models Look at when Answering Questions?**](https://arxiv.org/abs/2503.13891) | Explainability | 2024-03 | - |
| [**Benchmarking Large Vision-Language Models on Fine-Grained Image Tasks: A Comprehensive Evaluation**](https://arxiv.org/abs/2504.14988) | Fine-Grained Evaluation | 2024-04 | - |
| [**Object-Level Verbalized Confidence Calibration in Vision-Language Models**](https://arxiv.org/abs/2504.14848) | Calibration & Trust | 2024-04 | - |

## Security and Robustness

| Title | Venue | Date | Code | Project |
|-------|-------|------|------|---------|
| [**Effective Black-Box Multi-Faceted Attacks Breach Vision Large Language Model Guardrails**](https://arxiv.org/abs/2502.05772) | arXiv | 2024-02 | - | - |
| [**VLForgery Face Triad: Detection, Localization and Attribution via Multimodal Large Language Models**](https://arxiv.org/abs/2503.06142) | arXiv | 2024-03 | - | - |

## Future Challenges

| Challenge | Key Papers |
|-----------|------------|
| **Hallucination** | [Woodpecker](https://arxiv.org/pdf/2310.16045), [POPE](https://arxiv.org/pdf/2305.10355), [HallusionBench](https://arxiv.org/pdf/2310.14566) |
| **Reasoning** | [MM-RLHF](https://arxiv.org/pdf/2502.10391), [Visual CoT](https://arxiv.org/pdf/2403.16999.pdf), [Multimodal Chain-of-Thought](https://arxiv.org/pdf/2302.00923.pdf) |
| **Long-context Understanding** | [Long-VITA](https://arxiv.org/pdf/2502.05177), [LongLLaVA](https://arxiv.org/pdf/2409.02889), [LongVU](https://arxiv.org/pdf/2410.17434) |
| **Efficiency and Optimization** | [CuMo](https://arxiv.org/pdf/2405.05949), [EAGLE](https://arxiv.org/pdf/2408.15998), [LLaVA-Mini](https://arxiv.org/pdf/2501.03895) |
| **Alignment and Safety** | [OmniAlign-V](https://arxiv.org/abs/2502.18411), [MM-RLHF](https://arxiv.org/pdf/2502.10391), [LLaVA-RLHF](https://arxiv.org/abs/2309.14525) | 