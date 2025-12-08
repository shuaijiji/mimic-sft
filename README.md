
# Mimic SFT: A Powerful Framework for Large Model Fine-Tuning

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Mimic SFT is a comprehensive and user-friendly framework for fine-tuning large models, with a special focus on visual language models like Qwen-VL. Built on top of the [SWIFT (Scalable Wise Fine-Tuning)](https://github.com/modelscope/swift) library, Mimic SFT provides a streamlined and efficient workflow for training, evaluation, and inference.

## Key Features

*   **Multiple Fine-Tuning Methods**: Supports a wide range of fine-tuning methods, including LoRA, QLoRA, and full-parameter fine-tuning.
*   **Distributed Training**: Integrated with DeepSpeed ZeRO-2 for efficient distributed training on multiple GPUs.
*   **Vision-Language Model Support**: Optimized for fine-tuning vision-language models like Qwen-VL, with support for custom image and text datasets.
*   **Comprehensive Workflow**: Provides a complete end-to-end workflow, from data preparation to model training, evaluation, and inference.
*   **Extensible and Customizable**: The modular design allows for easy extension and customization to support new models, datasets, and fine-tuning methods.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/shuaijiji/mimic-sft.git
    cd mimic-sft
    ```

2.  **Create a virtual environment (optional but recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Quick Start

This example demonstrates how to perform LoRA fine-tuning on the `Qwen2.5-7B-Instruct` model using the `alpaca-gpt4` dataset.

```bash
# Visible GPUs
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type lora \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
              'AI-ModelScope/alpaca-gpt4-data-en#500' \
              'swift/self-cognition#500' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --model_name swift-robot
```

## Advanced Usage

### Distributed Training

Mimic SFT supports distributed training using DeepSpeed. The `train_qwen2_5_vl.sh` script provides an example of how to configure and run a distributed training job.

### Custom Datasets

To use your own dataset, you need to prepare a JSONL file where each line is a JSON object containing the required data fields. For vision-language models, you typically need to provide the image path and the corresponding text.

You can specify the column mapping using the `--columns` argument in the `swift sft` command. For example:

```bash
--columns '{"report": "text", "image_path": "images"}'
```

## Contributing

We welcome contributions from the community! If you would like to contribute to Mimic SFT, please follow these steps:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and commit them with a descriptive message.
4.  Push your changes to your fork.
5.  Create a pull request to the main repository.

## License

This project is licensed under the [Apache License 2.0](LICENSE).
qwen2.5-vl-7B在MIMIC-CXR上的sft训练

训练脚本
train_qwen2_5_vl.sh
