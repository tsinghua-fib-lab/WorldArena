### 1. Environment Setup

This module was developed and tested under **Python 3.10.19** with **cuda 12.8**. First, execute the following commands to create an environment named `worldarena_embodied` and activate it:

```bash
conda create -n worldarena_embodied python=3.10.19 -y
conda activate worldarena_embodied
```

Next, install PyTorch 2.9.1 (CUDA 12.8 version) compatible with NVIDIA GPUs to ensure the correct training and inference environment:

```bash
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
```

Then install flash-attn==2.8.3:

```bash
pip install flash-attn==2.8.3 --no-build-isolation
```

Finally, install the remaining dependencies (such as Transformers, OpenCV, etc.):

```bash
cd action_planner
pip install -r requirements.txt
```

### 2. Download Model Weights

As an example implementation, we fine-tune the [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) model on the RoboTwin 2.0 dataset to conduct the action planning task.

The fine-tuned video generation weights, together with the action planning module weights, are available at: https://huggingface.co/WorldArena/WorldArena/tree/main/models.

Additionally, it requires the following files:
- Download the [CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32) model and place it under the `./models` directory.
- Download the original [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) checkpoint.


The file structure is:

```bash
models/
├── wan_video/ 
│ └── wan_video.pt
├── wan_adjust_bottle/ 
│ └── wan_adjust_bottle.pt
└── wan_click_bell/ 
│ └── wan_click_bell.pt
├── clip-vit-base-patch32/ 
└── Wan2.2-TI2V-5B/ 
```


### 3. Quick Start

1.  **Edit Shell Script:**：
    *   File path：`./sripts/step1_prepare_latent_wan.sh` `./sripts/generate_metadata.py`
    *   Modify the variable `DATASET_PATH`, replacing its value with **your dataset path**.
    *   Modify the variable `OUTPUT_DIR`， replacing its value with the directory path where you want to store **output files**.

2.  **Edit Python Script**：
    *   File path：`step1_prepare_latent_wan.py`
    *   Modify the variable `ROOT`, replacing its value with **your dataset path**

After completing the above path replacements, run the preprocessing script.

```bash
python ./scripts/generate_metadata.py
bash ./sripts/step1_prepare_latent_wan.sh
```

Note: You can change the `allow_task` variable in `step1_prepare_latent_wan.py` to `adjust_bottle` or `click_bell` depending on the task you want to process. \
Training script path:

```bash
bash ./sripts/train_wan.sh
```

Evaluation script path:

```bash
bash ./sripts/evaluate_wan_single.sh
```
