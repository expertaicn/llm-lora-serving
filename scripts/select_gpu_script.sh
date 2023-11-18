#!/bin/bash

# 定义函数，接受 GPU 信息并返回选择的 GPU 索引
select_gpu() {
    gpu_info=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)

    # local gpu_info="$1"
    # echo "----- ${gpu_info} -----"
    # 将 GPU 信息按逗号分隔为数组
    # IFS=',' read -ra gpu_memory_free <<< "$gpu_info"
    readarray -t gpu_memory_free <<< "$gpu_info"


    local max_free_memory=0
    local max_second_memory=-1
    local selected_gpu=-1
    local selected_second_gpu=-1
    for i in "${!gpu_memory_free[@]}"; do
        local memory_free="${gpu_memory_free[i]}"
        # echo "${memory_free} ...."
        if ((memory_free > max_free_memory)); then
            max_second_memory=$max_free_memory
            max_free_memory=$memory_free
            selected_second_gpu=$selected_gpu
            selected_gpu=$i
        elif ((memory_free > max_second_memory)); then
            max_second_memory=$memory_free
            selected_second_gpu=$i
        fi
        
    done

    # 返回选定的 GPU 索引
    echo "$selected_gpu,$selected_second_gpu"
}

# 执行 nvidia-smi 命令，将输出保存到变量 nvidia_smi_output 中

# 调用函数并传递 GPU 信息作为参数
selected_gpu_index=$(select_gpu)
# 检查是否找到可用 GPU
if ((selected_gpu_index != -1)); then
    echo "Selected GPU: $selected_gpu_index"
else
    echo "No available GPU found."
fi