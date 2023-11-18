### build images
basepath=$(cd `dirname $0`; pwd)
echo "basepath: $basepath"
cd $basepath/..
image_name=llmserver:v1
container_name=llmserver_v1
# remove previous container and images
sudo docker stop ${container_name}
sudo docker rm ${container_name}
sudo docker rmi ${image_name}

# build images
sudo docker build  -f  Dockerfile -t ${image_name} .


### run a container baichuan1
#sudo docker run  --name ${container_name}  --gpus all  -d -p 8080:8000  -v /home/tzw/models/baichuan:/app/model/baichuang-chat-13b ${image_name}
#   - lora_name: common
#     lora_parent: /app/model/baichuan2_vllm/adapter/baichuan2_sft
#   - lora_name: douyin
#     lora_parent: /app/model/baichuan2_vllm/adapter/baichuan2_douyin
# /app/model/baichuan2_vllm
### run a container baichuan2
sudo docker run  --name ${container_name}  --gpus all  -d -p 8180:8711 --shm-size=10.24gb -v /home/tzw/models:/models ${image_name}



## login a container
sudo docker exec -it ${container_name} /bin/bash

# 当上述的命令有问题的时候可以使用下面的命令登陆，查看问题
# sudo docker run --rm -it --name ${container_name} --gpus all -p 8180:8000 \
# -v /run/tzw/models/baichuan2/baichuan2_vllm:/app/model/baichuan2_vllm \
# -v /data2/tzw/models/adapters:/app/model/baichuan2_vllm/adapter \
# ${image_name} /bin/bash
