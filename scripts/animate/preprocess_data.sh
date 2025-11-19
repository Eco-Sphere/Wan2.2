MODEL_PATH=/data2/test/zt/scripts/2025_Nov_Proj/wan2-animate/weights/Wan2.2-Animate-14B
CKPT_PATH="${MODEL_PATH}/process_checkpoint"

ANIMATE_ASSET_BASE_PATH="../../examples/wan_animate/animate"
ANIMATE_VIDEO_PATH="${ANIMATE_ASSET_BASE_PATH}/video.mp4"
ANIMATE_REFER_PATH="${ANIMATE_ASSET_BASE_PATH}/image.jpeg"
ANIMATE_SAVE_PATH="${ANIMATE_ASSET_BASE_PATH}/process_results"

REPLACE_ASSET_BASE_PATH="../../examples/wan_animate/replace"
REPLACE_VIDEO_PATH="${REPLACE_ASSET_BASE_PATH}/video.mp4"
REPLACE_REFER_PATH="${REPLACE_ASSET_BASE_PATH}/image.jpeg"
REPLACE_SAVE_PATH="${REPLACE_ASSET_BASE_PATH}/process_results"


mkdir -p ${ANIMATE_SAVE_PATH}
mkdir -p ${REPLACE_SAVE_PATH}

# Animate Preprocess
# python ../../wan/modules/animate/preprocess/preprocess_data.py \
#     --ckpt_path ${CKPT_PATH} \
#     --video_path ${ANIMATE_VIDEO_PATH} \
#     --refer_path ${ANIMATE_REFER_PATH} \
#     --save_path ${ANIMATE_SAVE_PATH} \
#     --resolution_area 1280 720 \
#     --retarget_flag \
#     --use_flux

# Replace Preprocess
python ../../wan/modules/animate/preprocess/preprocess_data.py \
    --ckpt_path ${CKPT_PATH} \
    --video_path ${REPLACE_VIDEO_PATH} \
    --refer_path ${REPLACE_REFER_PATH} \
    --save_path ${REPLACE_SAVE_PATH} \
    --resolution_area 1280 720 \
    --iterations 3 \
    --k 7 \
    --w_len 1 \
    --h_len 1 \
    --replace_flag