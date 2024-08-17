tssha() {
    gcloud compute tpus tpu-vm ssh "$1" --zone us-central2-b --worker=all --command "$2"
}
# echo "sudo docker run --net=host --privileged -t -d tpuvm $DOCKER_CMD"
tssha v4-64 "cd pytorch-tpu-transformers/ && \
git pull && \
./kill_tpu_docker.sh && \
sudo docker build -t tpuvm . --build-arg WANDB_API_KEY=$WANDB_API_KEY"

tssha v4-64 'sudo docker run \
-v /mnt/nfs_share/docker-cache:/root/.cache \
-v /mnt/nfs_share/docker-file:/root/files \
--net=host --privileged -t -d tpuvm'
