FROM us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.4.0_libtpu_3.10_tpuvm

# Allow overriding some training parameters at build time
ARG spmd_sharding_flag="--spmd_2d_sharding 2"
ARG global_batch_size=128
ARG libtpu_init_args=""
ARG WANDB_API_KEY=""
# Define a build-time argument with a default value
ARG WANDB_RUN_GROUP_DEFAULT="tpu-spmd-run-group"
# Use the build-time argument to set the environment variable
ENV WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-$WANDB_RUN_GROUP_DEFAULT}"

# Clone and install the SPMD-enabled fork of HF transformers
RUN pip install datasets accelerate evaluate scikit-learn wandb
RUN git clone -b llama2-google-next-training https://github.com/pytorch-tpu/transformers.git
RUN pip install git+file:///transformers datasets accelerate evaluate scikit-learn

# Copy relevant args to environment variables for use in CMD
ENV SPMD_SHARDING_FLAG="${spmd_sharding_flag}"
ENV GLOBAL_BATCH_SIZE="${global_batch_size}"
ENV LIBTPU_INIT_ARGS="${libtpu_init_args}"

ENV WANDB_API_KEY="${WANDB_API_KEY}"
ENV WANDB_RUN_GROUP="${WANDB_RUN_GROUP}"

COPY pytorch-tpu-transformers/examples/pytorch/run_clm.py \
    run_clm.py

# Run the training using the copied config file and specified sharding strategy
CMD python -u \
    /run_clm.py \
    --tokenizer_name beomi/Solar-Ko-Recovery-11B \
    --model_name_or_path beomi/Solar-Ko-Recovery-11B \
    --dataset_name maywell/korean_textbooks \
    --dataset_config_name claude_evol \
    --per_device_train_batch_size ${GLOBAL_BATCH_SIZE} \
    --num_train_epochs 3 \
    --do_train \
    --output_dir /root/files/beomi/Solar-Ko-Recovery-11B/ \
    --overwrite_output_dir \
    --save_strategy steps \
    --save_steps 10 \
    --logging_strategy steps \
    --logging_steps 1 \
    --remove_unused_columns no \
    --optim adafactor \
    --torch_dtype bfloat16 \
    --dataloader_drop_last yes \
    --spmd_grad_chkpt \
    --warmup_steps 200 \
    --bf16 \
    ${SPMD_SHARDING_FLAG}
