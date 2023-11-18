import logging
from pydantic import BaseModel
from llmserver.utils import ordered_yaml_load, ordered_yaml_dump
from typing import List
from pathlib import Path


class InferenceParam(BaseModel):
    name: str = "general"
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 5
    max_tokens: int = 100
    stop_list: List[str] = ["样例"]


class LoraInfo(BaseModel):
    lora_name: str = "fake"
    lora_parent: str = "fake"
    lora_model_path: str = "adapter_model.bin"
    lora_config_path: str = "adapter_config.json"


class ServerConfig(BaseModel):
    base_model_name_or_path: str
    server_port: int = 8711
    tensor_parallelism_size: int = 2
    lora_list: List[LoraInfo] = [LoraInfo()]
    inference_param_list: List[InferenceParam] = [InferenceParam()]

    @classmethod
    def load(cls, file_path):
        logging.info(f"begin load [{file_path}]")
        config_data = ordered_yaml_load(file_path)
        config = cls(**config_data)
        logging.info(
            f"done load [{file_path}] model path  is [{config.base_model_name_or_path}]"
        )
        return config


if __name__ == "__main__":
    config = ServerConfig(
        base_model_name_or_path="/home/tzw/models/baichuan2/baichuan2_13b_vllm"
    )
    ordered_yaml_dump(data=config.dict(), stream=open("../etc/dev/server.yml", "w"))
    file_path = Path(__file__).parent / ".." / "etc" / "dev" / "server_7b.yml"
    config = ServerConfig.load(file_path=file_path)
    print(config.base_model_name_or_path)
