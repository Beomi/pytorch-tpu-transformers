tssha() {
    gcloud compute tpus tpu-vm ssh "$1" --zone us-central2-b --worker=all --command "$2"
}
# echo "sudo docker run --net=host --privileged -t -d tpuvm $DOCKER_CMD"
tssha v4-32a "cd pytorch-tpu-transformers/ && \
git pull && \
./kill_tpu_docker.sh && \
sudo docker system prune -f && \
sudo docker build -t tpuvm . \
--build-arg WANDB_API_KEY=$WANDB_API_KEY \
--build-arg HF_TOKEN=$HF_TOKEN \
--build-arg CUR_TIME=$(date +%s)"

tssha v4-32a 'sudo docker run \
--net=host --privileged -t -d tpuvm'
