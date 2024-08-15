tssha() {
    gcloud compute tpus tpu-vm ssh "$1" --zone us-central2-b --worker=all --command "$2"
}

tssha v4-256 'screen -S trainer -X quit'
tssha v4-256 'screen -dmS trainer bash -c "cd pytorch-tpu-transformers && git pull && ./train.sh"'
