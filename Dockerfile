# FROM cuda-11.7-vllm-0.2.1-torch-2.0.1:v9
FROM cuda-12.1-vllm-0.2.6-torch-2.1.2:v9 

WORKDIR /app/llmserver
COPY . .
RUN mkdir -p /app/model

RUN pip install -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com -r requirements_prod.txt
EXPOSE 8000
CMD ["/bin/bash", "scripts/start_server.sh", "prod"]
#CMD ["/bin/bash", "scripts/start_api_prod.sh"]