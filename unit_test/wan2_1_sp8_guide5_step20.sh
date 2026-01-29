ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export ALGO=1
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false

export model_base=/tmp/weights/Wan2.1-I2V-14B-480P/

export OMP_NUM_THREADS=32

export WAN_MODEL_SCHEMA=Wan2.1

torchrun --nproc_per_node=8 ../generate.py \
--task i2v-A14B \
--ckpt_dir ${model_base} \
--size 768*432 \
--frame_num 57 \
--sample_steps 20 \
--t5_fsdp \
--sample_guide_scale 5.0 \
--ulysses_size 4 \
--cfg_size 2 \
--sample_solver unipc \
--image ../examples/i2v_input.JPG \
--prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
--base_seed 0
