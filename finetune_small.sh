#!/bin/bash
# reference: https://github.com/huggingface/transformers/tree/master/examples/pytorch/text-classification

export CUDA_VISIBLE_DEVICES=7

echo '[INFO]' $0 'starts...'
cd $(dirname "$0")
echo '[PWD]' $(pwd)
WORKDIR=$(pwd)

CKPT_PATH=${1:-./pretrain_results/saved_small_models/checkpoint-1000}
TASK=${2:-rte}
MAXL=${3:-128}
PERGPU_BS=${4:-32}
LR=${5:-5e-5}
EPOCH=${6:-4}
WARMUP_STEPS=${7:-100}
SEED=${8:-123}
EVAL_STEPS=${9}

# finetune the pretrained model on specific glue task
SAVE_DIR=./finetune_results/${TASK}/MAXL-${MAXL}_LR-${LR}_EPOCH-${EPOCH}_WARMUP-${WARMUP_STEPS}_BS-${PERGPU_BS}

if test -z $EVAL_STEPS; then
  python3 ./scripts/run_glue.py \
    --model_name_or_path ${CKPT_PATH} \
    --task_name ${TASK} \
    --max_seq_length ${MAXL} \
    --per_device_train_batch_size ${PERGPU_BS} \
    --learning_rate ${LR} \
    --num_train_epochs ${EPOCH} \
    --output_dir ${SAVE_DIR} \
    --warmup_steps ${WARMUP_STEPS} \
    --seed ${SEED} \
    --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --save_strategy epoch --logging_strategy epoch
else
  python3 ./scripts/run_glue.py \
    --model_name_or_path ${CKPT_PATH} \
    --task_name ${TASK} \
    --max_seq_length ${MAXL} \
    --per_device_train_batch_size ${PERGPU_BS} \
    --learning_rate ${LR} \
    --num_train_epochs ${EPOCH} \
    --output_dir ${SAVE_DIR} \
    --warmup_steps ${WARMUP_STEPS} \
    --seed ${SEED} \
    --do_train --do_eval --do_predict \
    --evaluation_strategy steps --save_strategy steps --logging_strategy steps \
    --eval_steps ${EVAL_STEPS} --save_steps ${EVAL_STEPS} --logging_steps ${EVAL_STEPS} 
fi

