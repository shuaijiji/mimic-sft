# Qwen3-VL-8B 微调训练配置说明

本项目基于 **Qwen3-VL-8B-Thinking** 多模态大模型，采用 **LoRA（Low-Rank Adaptation）** 高效微调策略，针对特定下游任务（如界面元素识别）进行适配。以下为完整训练参数说明。

---

## 一、核心与模型参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `--model` | `${MODEL_PATH}` | 指定基础模型路径（本地 `Qwen3-VL-8B-Thinking`） |
| `--dataset` | `${TRAIN_JSON}` | 训练数据集路径（JSON 格式） |
| `--val_dataset` | `${VAL_JSON}` | 验证数据集路径，用于监控泛化性能 |
| `--output_dir` | `${OUTPUT_DIR}` | 输出目录，保存 checkpoints、LoRA 权重、logs 等 |

---

## 二、训练策略与方法（LoRA + 多模态冻结）

| 参数 | 值 | 说明 |
|------|-----|------|
| `--train_type` | `lora` | 使用 LoRA 微调，仅训练低秩适配器，高效省显存 |
| `--lora_rank` | `32` | LoRA 适配器秩（rank），平衡拟合能力与开销（常用值） |
| `--lora_alpha` | `128` | LoRA 缩放因子，通常为 `rank × 2~4`，增强适配器影响 |
| `--lora_dropout` | `0.05` | 适配器层 Dropout 概率，轻度正则化防过拟合 |
| `--target_modules` | `all-linear` | 自动应用 LoRA 到所有线性层（推荐） |
| `--freeze_vit` | `true` | ❗冻结视觉编码器（ViT），避免破坏预训练视觉表征 |
| `--freeze_aligner` | `true` | ❗冻结视觉-语言对齐器（Aligner），仅微调语言部分 |

> ✅ 该设置显著减少可训练参数，适配 VL（视觉-语言）任务高效微调。

---

## 三、训练超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `--num_train_epochs` | `3` | 全量数据遍历 3 轮 |
| `--learning_rate` | `1e-4` | LoRA 常用学习率（0.0001），稳定且收敛快 |
| `--warmup_ratio` | `0.05` | 前 5% steps 线性学习率预热，提升训练初期稳定性 |
| `--per_device_train_batch_size` | `1` | 单卡 batch size = 1（因模型大，显存受限） |
| `--gradient_accumulation_steps` | `4` | 梯度累积 4 步 → **等效 batch size = 16**（4 GPUs × 1 × 4） |

---

## 四、性能与效率优化

| 参数 | 值 | 说明 |
|------|-----|------|
| `--dtype` | `bfloat16` | 混合精度训练（BF16），节省 50% 显存，提速显著，A100/H100 友好 |
| `--attn_impl` | `flash_attn` | 启用 FlashAttention，大幅加速 attention 计算，降低显存峰值 |
| `--padding_free` | `true` | 移除 padding tokens，跳过无效计算，提升吞吐 |
| `--deepspeed` | `zero2` | DeepSpeed ZeRO Stage 2：分片优化器状态 & 梯度，支持超大模型训练 |
| `--dataset_num_proc` | `4` | 4 进程并行预处理数据集（tokenization 等） |
| `--dataloader_num_workers` | `0` | 设为 0 避免 PicklingError（主进程加载数据） |

---

## 五、评估与保存策略

| 参数 | 值 | 说明 |
|------|-----|------|
| `--eval_steps` | `1000` | 每 1000 步在验证集评估一次 |
| `--save_steps` | `1000` | 每 1000 步保存 checkpoint |
| `--save_total_limit` | `3` | 最多保留 3 个最新 checkpoint，自动清理旧版本 |
| `--load_best_model_at_end` | `True` | 训练结束后自动加载**最佳模型** |
| `--metric_for_best_model` | `eval_loss` | 以验证集 loss 为评估指标 |
| `--greater_is_better` | `False` | `eval_loss` 越小越好 |

---

## 六、其他实用配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `--logging_steps` | `10` | 每 10 步打印训练日志（loss / lr 等） |
| `--max_length` | `2048` | 序列最大长度（含图像 token 和文本） |
| `--ddp_timeout` | `18000000` | DDP 超时时间（ms），避免大模型训练因卡顿中断 |
| `--ignore_args_error` | `True` | 忽略未知/冲突参数错误，提升脚本鲁棒性 |

---

## 💡 使用建议

- **显存优化**：`bf16 + flash_attn + deepspeed zero2 + freeze_vit/aligner` 组合可使 8B VL 模型在 4×A100（80GB）上稳定微调；
- **任务适配**：若你的任务是**固定位置按钮识别（如快手直播界面）**，建议在数据构造时加入位置偏置（如 bounding box prompt），提升泛化；
- **调试技巧**：首次运行可设 `--num_train_epochs 0.1` + `--logging_steps 1` 快速验证 pipeline。

> 项目由 **coolG（冀冀）** 维护 🌟  
> —— 硬核 · 聪明 · 幽默，持续探索 VL 落地边界
