DOCKER_CMD='\
pip3 install -U wandb && \
/root/.local/bin/wandb login $WANDB_API_KEY && \
python -u \
    /transformers/examples/pytorch/language-modeling/run_clm.py \
    --tokenizer_name beomi/Solar-Ko-Recovery-11B \
    --model_name_or_path beomi/Solar-Ko-Recovery-11B \
    --dataset_name maywell/korean_textbooks \
    --dataset_config_name claude_evol \
    --per_device_train_batch_size 256 \
    --num_train_epochs 3 \
    --do_train \
    --output_dir /root/files/beomi/Solar-Ko-Recovery-11B/ \
    --overwrite_output_dir \
    --save_strategy epoch \
    --logging_strategy steps \
    --logging_steps 1 \
    --remove_unused_columns no \
    --optim adafactor \
    --torch_dtype bfloat16 \
    --bf16 \
    --dataloader_drop_last yes \
    --preprocessing_num_workers 32 \
    --spmd_grad_chkpt \
'

tssha() {
    gcloud compute tpus tpu-vm ssh "$1" --zone us-central2-b --worker=all --command "$2"
}
# echo "sudo docker run --net=host --privileged -t -d tpuvm $DOCKER_CMD"
tssha v4-64 "cd pytorch-tpu-transformers && git pull && ./kill_tpu_docker.sh"

tssha v4-64 'sudo docker run \
-v /mnt/nfs_share/docker-cache:/root/.cache \
-v /mnt/nfs_share/docker-file:/root/files \
--net=host --privileged -t -d tpuvm "$DOCKER_CMD"'
