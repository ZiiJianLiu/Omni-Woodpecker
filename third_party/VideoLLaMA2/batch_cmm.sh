GPU_DEVICE=0,1,2,3,4,5,6,7
length=${#GPU_DEVICE}
n_gpu=$(((length+1)/2))
MODAL_TYPE=av
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATASET_PATH="${RELEASE_ROOT}/data/CMM"
OUTPUT_PATH=./results/test/evaluation_results_accelerate.json
DEVICE_MAP=auto
BATCH_SIZE=1
NUM_WORKERS=2
SAVE_INTERVAL=100
LOAD_8BIT=False
LOAD_4BIT=False
VERBOSE=False

CUDA_VISIBLE_DEVICES=$GPU_DEVICE \
accelerate launch --num_processes=$n_gpu --main_process_port 29506  \
eval_batch_cmm.py \
--category over-reliance_unimodal_priors \
--output-path ./results/my_output.json \
--modal-type av \

# CUDA_VISIBLE_DEVICES=$GPU_DEVICE \
# accelerate launch --num_processes=$n_gpu --main_process_port 29505  \
# eval_batch_cmm.py \
# --category over-reliance_unimodal_priors \
# --output-path ./results/my_output.json \
# --modal-type v \
