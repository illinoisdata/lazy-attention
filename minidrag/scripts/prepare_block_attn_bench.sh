#!/bin/bash
set -e

echo ">>> Step 1: Creating data directories"
mkdir -p block-attn-bench-datahub/raw_data/{tqa,nq,hqa,rag}

echo ">>> Step 2: Downloading 2WikiMultiHop dataset"

cd block-attn-bench-datahub/raw_data

# Check if git-lfs is installed
if ! command -v git-lfs &> /dev/null
then
    echo ">>> git-lfs not found."
else
    echo ">>> git-lfs already installed."
fi



git clone https://huggingface.co/datasets/xanhho/2WikiMultihopQA
ln -s 2WikiMultihopQA 2wiki
cd ..

echo ">>> Skipping Steps 3 and 4 (NQ/TQA) because of possible OOM "

# echo ">>> Step 3 : Cloning FiD and downloading NQ/TQA  "
# cd block-attn-bench-datahub
# git clone https://github.com/facebookresearch/FiD
# cd FiD
# bash get-data.sh
# cd ..

# echo ">>> Step 4: Creating symlinks for NQ and TQA"
# ln -s FiD/open_domain_data/TQA/test.json tqa/test.json
# ln -s FiD/open_domain_data/TQA/train.json tqa/train.json
# ln -s FiD/open_domain_data/NQ/test.json nq/test.json
# ln -s FiD/open_domain_data/NQ/train.json nq/train.json

echo ">>> Step 5: Downloading HotpotQA (HQA)"
cd hqa
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
cd ../..


echo ">>> Step 6: Constructing Block-Attn-Bench"
mkdir -p processed_data


python3 benchmarks/block-attn-bench/hqa.py \
    --eval_fp scripts/block-attn-bench-datahub/hqa/hotpot_dev_distractor_v1.json \
    --output_dir scripts/block-attn-bench-datahub/processed_data

# python3 benchmarks/block-attn-bench/nq.py \
#     --eval_fp block-attn-bench-datahub/nq/test.json \
#     --output_dir block-attn-bench-datahub/processed_data

# python3 benchmarks/block-attn-bench/tqa.py \
#     --eval_fp block-attn-bench-datahub/tqa/test.json \
#     --train_fp block-attn-bench-datahub/tqa/train.json \
#     --output_dir block-attn-bench-datahub/processed_data


if ! python3 -c "import fastparquet" &> /dev/null; then
    echo ">>> fastparquet not found. Installing via pip..."
    pip install fastparquet
else
    echo ">>> fastparquet is already installed."
fi

python3 benchmarks/block-attn-bench/2wiki.py \
    --dev_fp scripts/block-attn-bench-datahub/2wiki/dev.parquet \
    --train_fp scripts/block-attn-bench-datahub/2wiki/train.parquet \
    --output_dir scripts/block-attn-bench-datahub/processed_data

echo ">>> All done!"
