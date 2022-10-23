#!/bin/bash

export CUDA_VISIBLE_DEVICES=4,5

echo '[INFO]' $0 'starts...'
cd $(dirname "$0")
echo '[PWD]' $(pwd)
WORKDIR=$(pwd)

##############################################################
# available modes:
#    mlm (BERT)     - mask language modeling
#    taco (TACO)    - mlm + token-level contrastive
##############################################################

# read cmd params
MODE=${1:-mlm}
SAVE_DIR_NAME=${2:-saved_base_models}
PREPROCESS_N=${3:-8}
PER_GPU_BS=${4:-256}
GRAD_ACC=${5:-4}
MAX_SEQ_LEN=${6:-256}
MAX_STEPS=${7:-500000}
WARMUP_STEPS=${8:-10000}
SAVE_STEPS=${9:-10000}
FP16=${10:-True}
DATASET_PATH=${11:-processed_data}

# hyperparams for the `Token-level Contrastive` loss
NEG_K=${12:-50}
POS_WINDOW_SIZE=${13:-5}
TAU=${14:-0.07}
SIM_WEIGHT=${15:-1}
FOR_ALL_POS=${16:-True}
EXCEPT_MLM_POS=${17:-False}

# TACO variants:
# FOR_ALL_POS | EXCEPT_MLM_POS | effect
#     True    |      True      | 15% mlm + 85% tc
#     True    |      False     | 15% mlm + 100% tc  -> our "TACO" (as default)
#     False   |       N/A      | 15% mlm + 15% tc



# multi-workers processing does not support py version < 3.8, some fixs are needed
if [ $PREPROCESS_N == 1 ]; then
  echo "only single worker to preprocess data"
else
  echo "mutiple workers to preprocess data"
  # fix bugs for python with version < 3.8, e.g.
  #sudo mv -f connection.py /usr/local/lib/python3.7/dist-packages/multiprocess/
fi

if [ ${MODE} == "mlm" ]; then
  python3 scripts/run_mlm.py \
    --model_type bert \
    --config_name ./bert_base/bert_config.json \
    --tokenizer_name bert-base-uncased --preprocessing_num_workers ${PREPROCESS_N} \
    --do_train \
    --per_device_train_batch_size ${PER_GPU_BS} \
    --gradient_accumulation_steps ${GRAD_ACC} --max_steps ${MAX_STEPS} \
    --learning_rate 1e-4 --lr_scheduler_type linear --warmup_steps ${WARMUP_STEPS} \
    --mlm_probability 0.15 \
    --max_seq_length ${MAX_SEQ_LEN} \
    --weight_decay 0.01 \
    --output_dir ./pretrain_results/${SAVE_DIR_NAME} --save_strategy steps --save_steps ${SAVE_STEPS} \
    --logging_strategy steps --logging_steps 1000 \
    --fp16 ${FP16} --dataset_path ${DATASET_PATH}
elif [ ${MODE} == "taco" ]; then
  python3 scripts/run_mlm_with_contrastive.py \
    --model_type bert \
    --config_name ./bert_base/bert_config.json \
    --tokenizer_name bert-base-uncased --preprocessing_num_workers ${PREPROCESS_N} \
    --do_train \
    --per_device_train_batch_size ${PER_GPU_BS} \
    --gradient_accumulation_steps ${GRAD_ACC} --max_steps ${MAX_STEPS} \
    --learning_rate 1e-4 --lr_scheduler_type linear --warmup_steps ${WARMUP_STEPS} \
    --mlm_probability 0.15 \
    --max_seq_length ${MAX_SEQ_LEN} \
    --weight_decay 0.01 \
    --output_dir ./pretrain_results/${SAVE_DIR_NAME} --save_strategy steps --save_steps ${SAVE_STEPS} \
    --logging_strategy steps --logging_steps 1000 \
    --fp16 ${FP16} --dataset_path ${DATASET_PATH} \
    --tau ${TAU} --infonce_weight ${SIM_WEIGHT} --neg_k ${NEG_K} --pos_window_size ${POS_WINDOW_SIZE} --contrastive_for_all_pos ${FOR_ALL_POS} --contrastive_except_mlm_pos ${EXCEPT_MLM_POS}
else
  ehco "Valid mode must be choiced in ['mlm', 'taco']"
fi

