export ALGO=1
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false

export model_base=Wan2.2-I2V-A14B-Lightning

export OMP_NUM_THREADS=32

torchrun --nproc_per_node=8 ../generate.py \
--task i2v-A14B \
--ckpt_dir ${model_base} \
--size 1280*720 \
--frame_num 61 \
--sample_steps 4 \
--dit_fsdp \
--t5_fsdp \
--cfg_size 1 \
--sample_guide_scale 1.0 \
--ulysses_size 8 \
--sample_solver euler \
--image ../examples/i2v_input.JPG \
--prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
--base_seed 0
