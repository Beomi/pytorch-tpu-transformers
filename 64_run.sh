tssha() {
    gcloud compute tpus tpu-vm ssh "$1" --zone us-central2-b --worker=all --command "$2"
}

echo "[local] Killing TPU"
tssha v4-64 "sudo fuser -k /dev/accel0"

echo "[local] Removing TPU Lock"
tssha v4-64 "sudo rm -f /tmp/libtpu_lockfile"

echo "[local] Removing screens"
tssha v4-64 "killall screen"

tssha v4-64 'screen -S trainer -X quit'
tssha v4-64 'screen -dmSL trainer bash -c "cd pytorch-tpu-transformers && git pull && ./64_train.sh"'
