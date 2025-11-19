MODEL_PATH=/data2/test/zt/scripts/2025_Nov_Proj/wan2-animate/weights/Wan2.2-Animate-14B

ANIMATE_ASSET_BASE_PATH="../../examples/wan_animate/animate"
ANIMATE_VIDEO_PATH="${ANIMATE_ASSET_BASE_PATH}/video.mp4"
ANIMATE_REFER_PATH="${ANIMATE_ASSET_BASE_PATH}/image.jpeg"
ANIMATE_SAVE_PATH="${ANIMATE_ASSET_BASE_PATH}/process_results"

REPLACE_ASSET_BASE_PATH="../../examples/wan_animate/replace"
REPLACE_VIDEO_PATH="${REPLACE_ASSET_BASE_PATH}/video.mp4"
REPLACE_REFER_PATH="${REPLACE_ASSET_BASE_PATH}/image.jpeg"
REPLACE_SAVE_PATH="${REPLACE_ASSET_BASE_PATH}/process_results"

SRC_PATH=$REPLACE_SAVE_PATH

# export ASCEND_LAUNCH_BLOCKING=1

torchrun --nnodes 1 --nproc_per_node 8 ../../generate.py \
    --task animate-14B \
    --ckpt_dir ${MODEL_PATH} \
    --src_root_path ${SRC_PATH} \
    --refert_num 1 \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size 8 \
    --vae_parallel
