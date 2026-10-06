MODEL="facebook/opt-1.3b"

for num_layers in 20 12 2; do # 20 12 2
    for task in piqa wikitext; do # piqa wikitext
        CONFIG_NAME=""
        if [ "$task" = "wikitext" ]; then
            CONFIG_NAME="--dataset_config_name wikitext-2-raw-v1"
        fi

        CUDA_VISIBLE_DEVICES=0 accelerate launch \
            --mixed_precision=bf16 \
            offsite_tuning/run_clm.py \
            --model_name_or_path $MODEL \
            --dataset_name $task \
            $CONFIG_NAME \
            --per_device_train_batch_size 4 \
            --per_device_eval_batch_size 4 \
            --learning_rate 5e-5 \
            --num_train_epochs 5 \
            --lm_weight 1.0 \
            --kd_weight 0.0 \
            --seed 42 \
            --eval_steps 20 \
            --block_size 512 \
            --num_student_layers $num_layers \
            --student_l_pad 2 \
            --student_r_pad 2 \
            --train_module adapter \
            --save_module adapter \
            --no_teacher \
            --restart_training \
            --load_student emulators/${MODEL}/${num_layers}_2_2 \
            --output_dir mylogs/${MODEL}/${task}/ft_emulator/${num_layers}_2_2
    done
done

#====================== shutdown the machine ====================
#shutdown now


