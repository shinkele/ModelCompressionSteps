# ============================================================
# OPT-1.3B 模拟器（Emulator）蒸馏训练脚本
# 模型: OPT-1.3B (facebook/opt-1.3b)
# 学生层数: 20, 12, 2 (从24层中选取), 适配器层数: 2+2
# 训练方式: 仅训练学生层，LM loss + KD loss 联合优化
# 数据集: The Pile (预分词)
# ============================================================
MODEL="facebook/opt-1.3b"
bs=5
pad=2
# num_student_layers=20 # 20, 12, 2

for num_student_layers in 20 12 2; do
    CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --mixed_precision=bf16 \
    offsite_tuning/run_clm.py \
    --model_name_or_path $MODEL \
    --train_tokenized_dataset dataset/pile/opt_tokenized/00 \
    --val_tokenized_dataset dataset/opt_tokenized/wikitext-2-raw-v1 \
    --preprocessing_num_workers 10 \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size $bs \
    --learning_rate 1e-4 \
    --num_warmup_steps 50 \
    --lr_scheduler_type cosine \
    --num_train_epochs 1 \
    --lm_weight 1.0 \
    --kd_weight 30.0 \
    --seed 42 \
    --block_size 512 \
    --eval_steps 100 \
    --num_student_layers $num_student_layers \
    --student_l_pad $pad \
    --student_r_pad $pad \
    --output_dir emulators/${MODEL}/${num_student_layers}_${pad}_${pad}
done

#====================== shutdown the machine ====================
shutdown now