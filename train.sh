export PJRT_DEVICE=TPU
export XLA_USE_SPMD=1
export XLA_USE_BF16=1
export XLA_IR_DEBUG=1
export XLA_HLO_DEBUG=1

export LIBTPU_INIT_ARGS="--xla_enable_async_collective_permute=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true --xla_jf_spmd_threshold_for_windowed_einsum_mib=0"

export PROFILE_EPOCH=0
export PROFILE_STEP=3
export PROFILE_DURATION_MS=20000
export PROFILE_LOGDIR=/tmp/home/

python examples/pytorch/language-modeling/run_clm.py \
    --report_to wandb \
    --tokenizer_name beomi/Yi-Ko-34B \
    --dataset_name maywell/korean_textbooks \
    --dataset_config_name claude_evol \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --num_train_epochs 1 \
    --do_train \
    --output_dir /tmp/output \
    --overwrite_output_dir \
    --config_name ~/config.json \
    --save_strategy no \
    --logging_strategy no \
    --remove_unused_columns no \
    --optim adafactor \
    --torch_dtype bfloat16 \
    --dataloader_drop_last yes \
    --block_size 2048 \
    --spmd_2d_sharding 1 \
    --spmd_grad_chkpt >./logfile 2>&1
