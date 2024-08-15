tssha() {
    gcloud compute tpus tpu-vm ssh "$1" --zone us-central2-b --worker=all --command "$2"
}

echo "[local] Killing TPU"
tssha "sudo fuser -k /dev/accel0"

echo "[local] Removing TPU Lock"
tssha "sudo rm -f /tmp/libtpu_lockfile"

echo "[local] Removing screens"
tssha "killall screen"

tssha v4-256 'screen -S trainer -X quit'
tssha v4-256 'screen -dmSL trainer bash -c "cd pytorch-tpu-transformers && git pull && ./train.sh"'
