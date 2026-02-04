# instal msmodelslim
# git clone https://gitcode.com/Ascend/msmodelslim.git
# cd msmodelslim
# bash install.sh

# PREPARE STEP
BASE_MODEL=/weights/Wan2.2-I2V-A14B-Lightning
SAVE_PATH=/weights/Wan2.2-I2V-A14B-Lightning-w8a8c8-mxfp8
CONFIG_PATH=/codes/msmodelslim/lab_practice/wan2_2/wan2_2_w8a8c8_mxfp8_i2v.yaml
WAN2_2_SCRIPT_PATH=/codes/wan2.2

export PYTHONPATH=$WAN2_2_SCRIPT_PATH:$PYTHONPATH

cd $WAN2_2_SCRIPT_PATH


# export CPLUS_INCLUDE_PATH=/usr/include/c++/12/:/usr/include/c++/12/aarch64-openEuler-linux/:$CPLUS_INCLUDE_PATH

msmodelslim quant \
    --model_path $BASE_MODEL \
    --save_path $SAVE_PATH \
    --device npu \
    --model_type Wan2.2-I2V-A14B \
    --trust_remote_code True \
    --config_path $CONFIG_PATH

cd -