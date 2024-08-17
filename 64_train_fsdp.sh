export PJRT_DEVICE=TPU
# export XLA_USE_SPMD=1
# export XLA_USE_BF16=1
export XLA_IR_DEBUG=1
export XLA_HLO_DEBUG=1

export LIBTPU_INIT_ARGS="--xla_enable_async_collective_permute=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true --xla_jf_spmd_threshold_for_windowed_einsum_mib=0"

# export PROFILE_EPOCH=0
# export PROFILE_STEP=3
# export PROFILE_DURATION_MS=20000
# export PROFILE_LOGDIR=/tmp/home/

export HF_HUB_ENABLE_HF_TRANSFER=0

export MODEL_NAME='beomi/Solar-Ko-Recovery-11B'

/home/beomi/venv/bin/python -u examples/pytorch/xla_spawn.py \
    examples/pytorch/language-modeling/run_clm.py \
    --tokenizer_name $MODEL_NAME \
    --model_name_or_path $MODEL_NAME \
    --dataset_name maywell/korean_textbooks \
    --dataset_config_name claude_evol \
    --per_device_train_batch_size 32 \
    --num_train_epochs 2 \
    --do_train \
    --output_dir /tmp/output \
    --overwrite_output_dir \
    --save_strategy no \
    --logging_strategy steps \
    --logging_steps 1 \
    --optim adafactor \
    --block_size 1024 \
    --torch_dtype bfloat16 \
    --spmd_fsdp_sharding
