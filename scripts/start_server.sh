basepath=$(cd `dirname $0`; pwd)
echo "basepath: $basepath"

bin=python
py_file=${basepath}/../llmserver/api/serving.py
echo ${bin}
echo ${py_file}
env_type="$1"

cd $basepath
# 在其他脚本中调用 select_gpu 函数
source select_gpu_script.sh

# 调用 select_gpu 函数并传递 GPU 信息
selected_gpu_index=$(select_gpu)
echo "PYTHONPATH=${basepath}/../  &&  export CUDA_VISIBLE_DEVICES=${selected_gpu_index} &&  nohup ${bin}  ${py_file} -e $env_type > log_https 2>&1 &"
export PYTHONPATH=${basepath}/../  &&   export CUDA_VISIBLE_DEVICES=${selected_gpu_index} && ${bin}  ${py_file} -e $env_type > log_https 2>&1
~
~
~