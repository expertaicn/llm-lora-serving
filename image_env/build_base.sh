sudo docker build  -f dockerfile.base -t cuda-12.1-vllm-0.2.6-torch-2.1.2:v9 .

#########################################
#
# 如何将镜像推送到公司的hub里？
# sudo docker tag e8a3b08917dc <company_docker_hub>/cuda-11.7-vllm-0.13-torch-2.0.1:v9
# sudo docker push <company_docker_hub>/cuda-11.7-vllm-0.13-torch-2.0.1:v9
#########################################


#########################################
#
# 如何测试镜像
# sudo docker run -ti --gpus all -v /home/tzw:/test_run cuda-12.1-vllm-0.2.6-torch-2.1.2:v9 /bin/bash
# /home/tzw 下面有一个测试程序 test_torch.py
##########################################