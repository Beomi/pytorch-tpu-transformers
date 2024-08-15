# PyTorch-TPU-Transformers (GSPMD)

## Setup

```bash
tssha () {
        gcloud compute tpus tpu-vm ssh $1 --zone us-central2-b --worker=all --command $2
}

# TPU
tssha v4-256 '
sudo apt-get update
sudo apt-get install libopenblas-dev -y
pip3 install numpy
pip3 install typing-extensions
pip3 install torch~=2.4.0 torch_xla[tpu]~=2.4.0 -f https://storage.googleapis.com/libtpu-releases/index.html
'

tssha v4-256 '
git clone -b llama2-google-next-training https://github.com/pytorch-tpu/transformers.git
cd transformers
pip3 install git+file://$PWD
pip3 install datasets accelerate evaluate scikit-learn'

tssha v4-256 'curl https://huggingface.co/TheBloke/Llama-2-7B-fp16/raw/main/config.json --output ~/config.json'

tssha v4-256 '
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

cd transformers
python examples/pytorch/language-modeling/run_clm.py \
 --tokenizer_name hf-internal-testing/llama-tokenizer \
 --dataset_name wikitext \
 --dataset_config_name wikitext-2-raw-v1 \
 --per_device_train_batch_size 96 \
 --per_device_eval_batch_size 8 \
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
 --spmd_grad_chkpt
'
```
