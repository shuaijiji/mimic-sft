一、核心与模型参数
--model "${MODEL_PATH}"
作用：指定你要进行微调的基础模型。这里使用的是你本地路径下的 
Qwen3-VL-8B-Thinking
 模型。
--dataset "${TRAIN_JSON}"
作用：指定训练数据集的路径。
--val_dataset "${VAL_JSON}"
作用：指定验证数据集的路径。在训练过程中，会定期在这个数据集上进行评估，以监控模型的泛化能力。
--output_dir "${OUTPUT_DIR}"
作用：指定所有训练产物（如模型检查点、日志、LoRA 权重等）的保存目录。
二、训练策略与方法
--train_type lora
作用：指定微调方法为 LoRA (Low-Rank Adaptation)。这是一种高效的参数微调技术，它不改变原始模型的权重，而是通过训练一个小的“适配器”矩阵来适应新任务，极大地降低了显存消耗和计算量。
--lora_rank 32
作用：设置 LoRA 适配器矩阵的“秩”（rank）。这个值越大，LoRA 适配器的参数就越多，模型的拟合能力可能更强，但也会增加显存占用。
32
 是一个比较常用的值，在性能和效率之间取得了很好的平衡。
--lora_alpha 128
作用：LoRA 的缩放因子。通常设置为 
lora_rank
 的 2 到 4 倍。它像一个“增强器”，用来调整 LoRA 适配器对模型原始输出的影响程度。
--lora_dropout 0.05
作用：在 LoRA 适配器层上应用的 Dropout 概率。这是一种正则化技术，通过随机“丢弃”一部分神经元来防止模型过拟合，增强其泛化能力。
0.05
 是一个比较温和的设置。
--target_modules all-linear
作用：指定将 LoRA 适配器应用到模型的哪些模块上。
all-linear
 是一个非常方便的设置，它会自动找出模型中所有的线性层（Linear layers，通常是权重最集中的地方）并应用 LoRA。
--freeze_vit true
 和 
--freeze_aligner true
作用：这是针对多模态模型 
Qwen-VL
 的关键优化。它会冻结模型的视觉部分（ViT, Vision Transformer）和连接视觉与语言的对齐器（Aligner）的权重，只训练语言模型部分的 LoRA 适配器。这极大地减少了需要训练的参数量，并保留了模型强大的视觉特征提取能力。
三、训练超参数
--num_train_epochs 3
作用：指定整个训练数据集将被完整地遍历 3 次。
--learning_rate 1e-4
作用：设置学习率，即模型权重在每次更新时调整的幅度。
1e-4
 (即 0.0001) 是 LoRA 微调中一个常见且有效的学习率。
--warmup_ratio 0.05
作用：设置学习率预热（Warmup）的比例。在训练开始的最初 5% 的步骤里，学习率会从一个很小的值逐渐增加到设定的 
1e-4
。这种“热身”有助于训练过程在早期更加稳定。
--per_device_train_batch_size 1
作用：指定每张 GPU 在一次前向传播中处理的样本数量。因为 
Qwen3-VL
 模型很大，设置为 
1
 可以最大限度地减少显存占用。
--gradient_accumulation_steps 4
 (根据你的脚本计算得出)
作用：梯度累积步数。虽然我们每次只处理 1 个样本，但计算出的梯度会先累积 4 次，然后再统一进行一次模型权重更新。这使得有效批次大小 (Effective Batch Size) 达到了 
1 (per_device) * 4 (GPUs) * 4 (accumulation) = 16
。这有助于稳定训练过程，获得更好的性能。
四、性能与效率优化
--dtype bfloat16
作用：使用 
bfloat16
（脑浮点16）混合精度进行训练。相比于标准的 
float32
，它能将显存占用减少近一半，并大幅提升计算速度，同时在现代 GPU（如 A100）上保持了很好的数值稳定性。
--attn_impl flash_attn
作用：使用 FlashAttention 库来优化注意力（Attention）机制的计算。这是一个革命性的优化，能极大地提升训练速度并减少显存消耗。
--padding_free true
作用：启用一个优化，可以移除输入序列中的填充（padding）部分，从而减少不必要的计算，提升训练效率。
--deepspeed zero2
作用：使用 DeepSpeed ZeRO Stage 2 进行分布式训练。这是一种强大的显存优化技术，它会将模型的优化器状态和梯度分散到各个 GPU 上，从而让单个 GPU 能放下更大的模型。
--dataset_num_proc 4
作用：使用 4 个 CPU 核心来并行处理数据集的预处理工作（如 tokenization），加快数据准备速度。
--dataloader_num_workers 0
作用：设置数据加载器的工作进程数。我们之前为了解决 
PicklingError
 将其设置为了 
0
，意味着数据加载在主进程中进行。
五、评估与保存策略
--eval_steps 1000
作用：每训练 1000 个步骤，就使用验证集进行一次评估。
--save_steps 1000
作用：每训练 1000 个步骤，就保存一次模型的检查点（checkpoint）。
--save_total_limit 3
作用：最多只保留最近的 3 个检查点，旧的会被自动删除，以节省磁盘空间。
--load_best_model_at_end True
作用：当训练全部结束后，自动加载在所有评估中表现最好的那个检查点作为最终模型。
--metric_for_best_model eval_loss
作用：指定衡量“最佳模型”的指标是验证集上的损失（
eval_loss
）。
--greater_is_better False
作用：告诉框架，
eval_loss
 这个指标是越小越好。
六、其他
--logging_steps 10
作用：每 10 个训练步骤，就在控制台打印一次训练日志（如 loss, learning rate 等）。
--max_length 2048
作用：设置模型能处理的最大序列长度。
--ddp_timeout 18000000
作用：设置分布式数据并行（DDP）的超时时间，延长这个时间可以避免在处理大模型或慢节点时因超时而中断。
--ignore_args_error True
作用：忽略一些不兼容或未知的参数错误，增加脚本的兼容性。
