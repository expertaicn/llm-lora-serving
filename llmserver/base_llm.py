from abc import ABC, abstractmethod
from pathlib import Path
from pydantic import BaseModel
from typing import Optional, List
import logging
import time


class GPUMemmoryInfo(BaseModel):
    whole_memory: float = 0
    free_memory: float = 0


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
            logging.info(f"函数 {new_name} 生成token {token_length} 执行时间：{elapsed_time_formatted} 秒")

            return result

        return wrapper

    return decorator


class BaseLLM(ABC):
    @abstractmethod
    def load(self, model_path: str, gpu_mem_info: Optional[GPUMemmoryInfo] = None) -> bool:
        """load model

        Args:
            model_path (str):
            gpu_mem_info (Optional[GPUMemmoryInfo], optional): . Defaults to None.

        Returns:
            bool: True, model load successfully, otherwise False
        """
        pass

    @abstractmethod
    def reason(self, prompt_str: str, inference_config_path: Path, inference_task_name: str) -> Optional[str]:
        """不同的推理模块有不同的推理配置.
          inference_config_path是一个文件，其内容的格式可以参考：../../prompts/generate/v1/inference_config.json
          inference_task_name 是为了方便模块将推理配置缓存起来，不用每次推理的时候都要加载文件。

        Args:
            prompt_str (str):
            inference_config_path (Path):
            inference_task_name (str):

        Returns:
            str: 大模型的返回的字符串, 没有进行加工过的字符串。None, 内部有错误发生
        """
        pass

    def batch_reason(
        self,
        prompt_str_list: List[str],
        inference_config_path: Path,
        inference_task_name: str,
        stop_list=[],
        lora_name=None,
    ) -> List[str]:
        results = []
        for prompt_str in prompt_str_list:
            response = self.reason(
                prompt_str=prompt_str,
                inference_config_path=inference_config_path,
                inference_task_name=inference_task_name,
            )
            # if response is not None:
            results.append(response)
        return results

    def multi_reason(
        self,
        prompt_str_list: List[str],
        inference_config_path: Path,
        inference_task_name: str,
        n: int = 0,
        stop_list=[],
        lora_name=None,
    ) -> List[List[str]]:
        pass
