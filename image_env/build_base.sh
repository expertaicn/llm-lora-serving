sudo docker build  -f dockerfile.base -t cuda-11.7-vllm-0.2.1-torch-2.0.1:v9 .

#########################################
#
# 如何将镜像推送到公司的hub里？
# sudo docker tag e8a3b08917dc <company_docker_hub>/cuda-11.7-vllm-0.13-torch-2.0.1:v9
# sudo docker push <company_docker_hub>/cuda-11.7-vllm-0.13-torch-2.0.1:v9
#########################################
