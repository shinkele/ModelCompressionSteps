MODEL="gpt2-xl"
#num_student_layers_list="40 36 32 28 24 20 16 12 8 4"
bs=1

#================== Test setting ====================
num_student_layers_list="40"
#====================================================

pad=4
for num_student_layers in $num_student_layers_list; do
    CUDA_VISIBLE_DEVICES="0,1" accelerate launch \
        --mixed_precision=bf16 --multi_gpu \
        --num_processes=2 \
        --num_machines=1 \
        --dynamo_backend=no \
        offsite_tuning/run_clm.py \
        --model_name_or_path $MODEL \
        --dataset_name piqa \
        --per_device_train_batch_size $bs \
        --per_device_eval_batch_size $bs \
        --learning_rate 1e-4 \
        --num_train_epochs 1 \
        --num_warmup_steps 100 \
        --lr_scheduler_type cosine \
        --lm_weight 1.0 \
        --kd_weight 0.0 \
        --no_save_model \
        --seed 42 \
        --block_size 512 \
        --eval_steps 20 \
        --num_student_layers $num_student_layers \
        --student_l_pad ${pad} \
        --student_r_pad ${pad} \
        --train_module adapter \
        --output_dir mylogs/layerdrop/${MODEL}/piqa/${num_student_layers}_${pad}_${pad}
done
