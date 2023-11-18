from abc import ABC, abstractmethod
from pathlib import Path
from pydantic import BaseModel
from typing import Optional, List
import logging
import time
import glob
import json
from llmserver.config import LoraInfo, InferenceParam, ServerConfig


def timing_decorator(new_name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            elapsed_time_formatted = "{:.2f}".format(elapsed_time)
            token_length = 0
            if result is not None:
                token_length = len(result)
            logging.info(
                f"函数 {new_name} 生成token {token_length} 执行时间：{elapsed_time_formatted} 秒"
            )

            return result

        return wrapper

    return decorator


class BaseLLMServer(ABC):
    def build(self, server_config: ServerConfig) -> bool:
        self.etc_dir = Path(__file__).parent / ".." / "etc" / "inference"
        file_pattern = "*.json"
        example_files = glob.glob(f"{self.etc_dir}/{file_pattern}")
        additional_inference_params: List[InferenceParam] = []
        for file_name in example_files:
            inference_type_name = file_name.split("/")[-1].split(".")[0]
            print(file_name)
            content = open(file_name, "r").read()
            print(content)
            data = json.loads(open(file_name, "r").read())
            temperature = data.get("temperature", 1.0)
            top_p = data.get("top_p", 1.0)
            top_k = data.get("top_k", -1)
            max_tokens = data.get("max_new_tokens", 16)
            stop_list = data.get("stop_list", [])
            inference_param = InferenceParam(
                name=inference_type_name,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                stop_list=stop_list,
            )
            additional_inference_params.append(inference_param)
        # 构造lora dict
        self.lora_dict = {
            item.lora_name: {
                "model_full_path": Path(item.lora_parent) / item.lora_model_path,
                "config_full_path": Path(item.lora_parent) / item.lora_config_path,
            }
            for item in server_config.lora_list
        }
        # 构造inference dict
        self.inference_type_dict = {
            item.name: item for item in additional_inference_params
        }
        for item in server_config.inference_param_list:
            self.inference_type_dict[item.name] = item

    @abstractmethod
    def reason(
        self,
        prompt_str: str,
        inference_type_name: str,
        inference_param: InferenceParam,
        lora_name: str,
    ) -> Optional[str]:
        pass

    def batch_reason(
        self,
        prompt_str_list: List[str],
        inference_type_name: str,
        inference_param: InferenceParam,
        lora_name=None,
    ) -> List[str]:
        results = []
        for prompt_str in prompt_str_list:
            response = self.reason(
                prompt_str=prompt_str,
                inference_type_name=inference_type_name,
                inference_param=inference_param,
                lora_name=lora_name,
            )
            # if response is not None:
            results.append(response)
        return results

    def multi_reason(
        self,
        prompt_str_list: List[str],
        inference_type_name: str,
        inference_param: InferenceParam,
        n: int = 0,
        stop_list=[],
        lora_name=None,
    ) -> List[List[str]]:
        pass
