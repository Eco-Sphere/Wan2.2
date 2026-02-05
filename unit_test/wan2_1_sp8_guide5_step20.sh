ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

export HCCL_TOPO_FILE_PATH=/usr/local/Ascend/driver/topo/a5/card_4p_mesh.json
export HCCL_BUFFSIZE=1024
export HCCL_CONNECT_TIMEOUT=120
export HCCL_EXEC_TIMEOUT=120
export HCCL_ALGO=level0:fullmesh

export ALGO=3
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false

export model_base=/home/data1/Wan2.1-I2V-14B-480P/
export quant_dit_path=/home/data1/Wan2.1-I2V-14B-480P-QuantFA/i2v_low_noise_model
# export quant_dir=/home/data1/Wan2.1-I2V-14B-480P-w8a8c8-mxfp8/i2v_low_noise_model


export OMP_NUM_THREADS=32

export WAN_MODEL_SCHEMA=Wan2.1

export PRECISION=1

torchrun --nproc_per_node=4 ../generate.py \
--task i2v-A14B \
--ckpt_dir ${model_base} \
--size 768*432 \
--frame_num 57 \
--sample_steps 20 \
--sample_guide_scale 5.0 \
--ulysses_size 4 \
--cfg_size 1 \
--sample_solver unipc \
--image ../examples/i2v_input.JPG \
--prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
--base_seed 0 \
--quant_dit_path $quant_dit_path \
