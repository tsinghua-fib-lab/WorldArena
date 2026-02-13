# WorldArena: A Unified Benchmark for Evaluating Perception and Functional Utility of Embodied World Models

</div>

<div align="center">

<a href="https://world-arena.ai/">
  <img src="https://img.shields.io/badge/Website-WorldArena-2563eb?style=for-the-badge&logo=googlechrome&logoColor=white">
</a>

<a href="https://arxiv.org/abs/2602.08971">
  <img src="https://img.shields.io/badge/Paper-arXiv-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white">
</a>

<a href="https://huggingface.co/spaces/WorldArena/WorldArena">
  <img src="https://img.shields.io/badge/Leaderboard-HuggingFace-ffcc00?style=for-the-badge&logo=huggingface&logoColor=ffcc00">
</a>

</div>


## Table of Contents

- [Updates](#-updates)
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Video Quality Evaluation](#-video-quality-evaluation)
- [Embodied Task Evaluation](#-embodied-task-evaluation)
- [Leaderboard](#-leaderboard)
- [Submission](#-submission)
- [Human Evaluation](#-human-evaluation)
- [Citation](#-citation)



## 📢 Updates

- [2026/02/13] Initial release.
- [2026/02/13] Leaderboard release.


## 🔍 Overview

WorldArena is a unified benchmark designed to systematically evaluate embodied world models across both **perceptual** and **functional** dimensions. WorldArena assesses models through **(1) video perception quality**, measured with sixteen metrics across six sub-dimensions; **(2) embodied task functionality**, which evaluates world models as synthetic data engines, policy evaluators, and action planners; **(3) human evaluations**, including overall quality, physics adherence, instruction following and head-to-head win rate. Furthermore, we propose **EWMScore**, a holistic metric integrating multi-dimensional performance into a single interpretable index. This work provides a framework for tracking progress toward truly functional world models in embodied AI.


## 📦 Dataset

This project uses a subset of the **RoboTwin 2.0** dataset. RoboTwin 2.0 is a simulation framework and benchmark for **bimanual robotic manipulation**, providing expert trajectories across a diverse set of manipulation tasks and supporting structured domain randomization for robust evaluation.

### Subset used in this project
- **50 tasks** (Clean-50 setting)
- **50 episodes per task**
- **Train/Test split (per task):** **40** episodes for training, **10** episodes for testing
- **Robot embodiment:** **Aloha-AgileX** (as indicated by `aloha-agilex_clean_50`)

### Data organization (per task)
A typical task directory follows the structure below (example: `adjust_bottle/aloha-agilex_clean_50/aloha-agilex_clean_50/`):

- `actions/` — low-level action sequences
- `video/` — episode videos
- `states/` — robot states / per-timestep metadata
- `instructions/` (+ optional `instructions_1`, `instructions_2`) — natural language task instructions / variants

### Links
- Official website & documentation: https://robotwin-platform.github.io/  
- Official repository: https://github.com/RoboTwin-Platform/RoboTwin  
- Dataset release (Hugging Face): https://huggingface.co/datasets/TianxingChen/RoboTwin2.0


## 🎬 Video Quality Evaluation
<div align="center">

<img src="assets/video_eval.png" width="85%">

</div>

## 🤖 Embodied Task Evaluation

<div align="center">

<img src="assets/task_eval.png" width="85%">

</div>

Please refer to [embodied task]() for implementation.

## 🏆 Leaderboard

The official WorldArena leaderboard is hosted on HuggingFace: [![Leaderboard](https://img.shields.io/badge/Leaderboard-HuggingFace-2D2D2D?style=flat&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/spaces/WorldArena/WorldArena). It provides standardized evaluation results across video perception quality, embodied task functionality, and the unified EWMScore. We welcome community submissions to benchmark new embodied world models under a fair and reproducible protocol. Join us in advancing truly functional world models for embodied AI.


## 📤 Submission
Coming soon.

## 👥 Human Evaluation
Be part of shaping the future of embodied world models!  👉 **Start here:**  [Human Evaluation](https://sd64n7jjtvotb9m1apn80.apigateway-cn-beijing.volceapi.com/)

We invite you to participate in our human evaluation by providing your judgment about generated videos — it only takes a few minutes. Your feedback helps us uncover hidden failure cases and align automated metrics with real human perception. Every contribution strengthens a more trustworthy and community-driven leaderboard.


## 🙌 Acknowledgement

We acknowledge [RoboTwin 2.0](https://robotwin-platform.github.io/) for providing the dataset and simulation platform support that enables embodied task evaluation.  

We thank [VPP](https://github.com/roboterax/video-prediction-policy) for providing the IDM framework used in our embodied action planning implementation.

For video quality evaluation, WorldArena references and partially builds upon the code implementations of the following projects:

- [VBench](https://github.com/Vchitect/VBench)  
- [EWMBench](https://github.com/AgibotTech/EWMBench)  
- [WorldScore](https://github.com/haoyi-duan/WorldScore)  
- [EvalCrafter](https://github.com/evalcrafter/EvalCrafter)  
- [JEDI](https://github.com/oooolga/JEDi)




## 📖 Citation

