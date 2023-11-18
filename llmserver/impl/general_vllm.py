from vllm import LLM, SamplingParams
from abc import ABCMeta
from pathlib import Path
import json
import logging
from typing import Optional, List
from llmserver.custom_types import SingletonType
from llmserver.base_llm_server import BaseLLM, GPUMemmoryInfo, timing_decorator
import threading
import copy
import torch
from datetime import datetime
from llmserver.config import LoraInfo


def lora_decorator(func):
    def wrapper(self, *args, **kwargs):
        with self.lock:
            # 提取 merge 和 name 参数
            lora_name = kwargs.get("lora_name", None)
            if lora_name is None or lora_name == "":
                logging.info("using original model .....")
                result = func(self, *args, **kwargs)
                return result
            lora_info_dict = self.lora_dict.get(lora_name, None)
            # logging.info(str(self.lora_dict) + " ------!!!!------ ")
            if lora_info_dict is None:
                logging.error(f"try to use {lora_name}, but no config")
                result = func(self, *args, **kwargs)
                return result
            lora_model_path = lora_info_dict["model_full_path"]
            lora_config_path = lora_info_dict["config_full_path"]
            logging.info(f"【 add lora switch start 】 {datetime.now()}")
            # TODO multi gpus adaptation
            lora_state = torch.load(lora_model_path, map_location="cuda:0")
            lora_config = Lora_Config(lora_config_path)
            adapt_lora(self.model, lora_state, lora_config, merge=True)
            logging.info(f"【 add lora switch end 】 {datetime.now()}")
            # 修改inference_config参数的值
            # kwargs["name"] = "Modified_" + name
            # 调用原始函数
            result = func(self, *args, **kwargs)
            # unmerge lora
            # Attention: the lora to unmerge should be the one merged
            logging.info(f"【 release lora switch start 】 {datetime.now()}")
            adapt_lora(self.model, lora_state, lora_config, merge=False)
            logging.info(f"【 release lora switch end 】 {datetime.now()}")
            return result

    return wrapper


class VLLMMetaClass(ABCMeta, SingletonType):
    pass


class GeneralVLLM(BaseLLM, metaclass=VLLMMetaClass):
    @classmethod
    def instance(cls):
        return GeneralVLLM()

    def __init__(self):
        self.model = None
        # key is str, value is SamplingParams
        self.inference_config_dict = {}
        # baichuan13b需要33g的空间
        self.model_size_table = {
            "baichuan": 33 * 1000,
            "baichuang": 33 * 1000,
            "Baichuan-13B-Chat": 33 * 1000,
            "baichuang-chat-13b": 33 * 1000,
            "baichuan2_vllm": 33 * 1000,
            "baichuan2": 33 * 1000,
            "baichuan2_sft": 33 * 1000,
            "baichuan2_title": 33 * 1000,
            "baichuan2_hunjian": 33 * 1000,
            "hallucination_baichuan13b-v2-4-13-vllm-adapter": 33 * 1000,
            "baichuan2-7b-base_vllm": 20 * 1000,
        }
        self.lock = threading.Lock()  # 创建一个线程锁

    def load(
        self,
        model_path: str,
        gpu_mem_info: Optional[GPUMemmoryInfo] = None,
        lora_info_list: List[LoraInfo] = [],
    ) -> bool:
        # 计算合适的给gpu_memory_utilization
        model_name = model_path.split("/")[-1]
        print(model_name)
        model_size = self.model_size_table.get(model_name, 33 * 1000)
        if model_size is None:
            logging.error(f"unknow the size of the model: {model_path}")
            return False
        if gpu_mem_info is None or gpu_mem_info.whole_memory <= 0:
            logging.error(
                "do not know the gpu info[whole memory], cannot assigin proper gpu memrory utilization"
            )
            return False
        gpu_memory_utilization = float(model_size) / gpu_mem_info.whole_memory
        print(model_size, gpu_mem_info.whole_memory, "------")
        self.model = LLM(
            dtype="float16",  # TODO when use lora, should uncomment this line
            model=model_path,
            trust_remote_code=True,
            tokenizer_mode="slow",  # tricky, some model does not support fast tokenizer
            gpu_memory_utilization=gpu_memory_utilization,
        )
        self.lora_dict = {
            item.lora_name: {
                "model_full_path": Path(item.lora_parent) / item.lora_model_path,
                "config_full_path": Path(item.lora_parent) / item.lora_config_path,
            }
            for item in lora_info_list
        }
        return True

    @timing_decorator(new_name="vllm")
    def reason(
        self,
        prompt_str: str,
        inference_config_path: Path,
        inference_task_name: str,
        stop_list=[],
        lora_name=None,
    ) -> str:
        results = self.batch_reason(
            [prompt_str],
            inference_config_path=inference_config_path,
            inference_task_name=inference_task_name,
            stop_list=stop_list,
            lora_name=lora_name,
        )
        if len(results) == 0:
            return None
        else:
            return results[0]

    @lora_decorator
    def batch_reason(
        self,
        prompt_str_list: List[str],
        inference_config_path: Path,
        inference_task_name: str,
        stop_list=[],
        lora_name=None,
    ) -> List[str]:
        if inference_task_name not in self.inference_config_dict:
            config_content = open(inference_config_path, "r").read().strip()
            data = json.loads(config_content)
            temperature = data.get("temperature", 1.0)
            top_p = data.get("top_p", 1.0)
            top_k = data.get("top_k", -1)
            max_tokens = data.get("max_new_tokens", 16)
            if len(stop_list) == 0:
                stop_list = data.get("stop_list", [])
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                stop=stop_list,
            )
            logging.debug(f"temperature: {temperature}, top_p: {top_p}, top_k: {top_k}")
            self.inference_config_dict[inference_task_name] = sampling_params
        sampling_params = self.inference_config_dict[inference_task_name]

        if len(stop_list) != 0:
            sampling_params_new = copy.deepcopy(sampling_params)
            sampling_params_new.stop = stop_list
            sampling_params = sampling_params_new
        results = []
        outputs = self.model.generate(prompt_str_list, sampling_params)
        for output in outputs:
            generated_text = output.outputs[0].text
            results.append(generated_text)
            logging.debug(f"output: [{generated_text}]")
        return results

    @lora_decorator
    def multi_reason(
        self,
        prompt_str_list: List[str],
        inference_config_path: Path,
        inference_task_name: str,
        n: int = 0,
        stop_list=[],
        lora_name=None,
    ) -> List[List[str]]:
        if inference_task_name not in self.inference_config_dict or n != 0:
            config_content = open(inference_config_path, "r").read().strip()
            data = json.loads(config_content)
            if n == 0:
                n = data.get("n", 1)
            temperature = data.get("temperature", 1.0)
            top_p = data.get("top_p", 1.0)
            top_k = data.get("top_k", -1)
            max_tokens = data.get("max_new_tokens", 16)
            if len(stop_list) == 0:
                stop_list = data.get("stop_list", [])
            sampling_params = SamplingParams(
                n=n,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                stop=stop_list,
            )
            logging.debug(f"temperature: {temperature}, top_p: {top_p}, top_k: {top_k}")
            self.inference_config_dict[inference_task_name] = sampling_params
        sampling_params = self.inference_config_dict[inference_task_name]

        if len(stop_list) != 0:
            sampling_params_new = copy.deepcopy(sampling_params)
            sampling_params_new.stop = stop_list
            sampling_params = sampling_params_new
        batch_results = []
        batch_outputs = self.model.generate(prompt_str_list, sampling_params)
        for outputs in batch_outputs:
            results = []
            for output in outputs.outputs:
                generated_text = output.text
                results.append(generated_text)
                logging.debug(f"output: [{generated_text}]")
            batch_results.append(results)
        return batch_results


def create_vllm(model_path, gpu_memory_need=33, lora_info_list: List = []):
    from llmserver.utils import get_available_gpu_with_free_memory
    import os

    gpu_index, toal_mem, free_mem = get_available_gpu_with_free_memory(
        gpu_memory_need, exclude_list=[], default_gpu_index=None
    )
    if gpu_index is None:
        logging.error(f"no available gpu have free memory {gpu_memory_need}")
        return
    logging.info(
        f"to use gpu: {gpu_index}, total_mem: {toal_mem}, free_mem: {free_mem}"
    )
    mem_info = GPUMemmoryInfo(whole_memory=toal_mem, free_memory=free_mem)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    # GeneralVLLM().instance().load(model_path, gpu_mem_info=mem_info)
    logging.debug(f"【 before load 】 {datetime.now()}")
    GeneralVLLM().instance().load(
        model_path, gpu_mem_info=mem_info, lora_info_list=lora_info_list
    )
    logging.debug(f"【 after load 】 {datetime.now()}")
    return GeneralVLLM().instance()


def transpose(weight, fan_in_fan_out):
    return weight.T if fan_in_fan_out else weight


def switch_lora(
    base_model, lora_weights, r, lora_alpha, fan_in_fan_out=False, merge=True
):
    # TODO check if the lora to unmerge is the one merged
    is_merged = getattr(base_model, "is_merged", False)
    assert is_merged != merge, (
        f"{is_merged} != {merge}: " "merge only when unmerged, vice versa"
    )
    base_model_weights = [(n, p) for n, p in base_model.named_parameters()]
    scaling = lora_alpha / r
    logging.info(f"Lora configs: alpha={lora_alpha}, r={r}, scaling={scaling}")
    lora_weights = {
        k.replace("base_model.model.", ""): v for k, v in lora_weights.items()
    }
    switched_keys = set()
    # base_model_weight_key v.s. lora_weight_key
    module_map = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "W_pack": ["W_pack"],
        "o_proj": ["o_proj"],
        "down_proj": ["down_proj"],
    }
    for name, param in base_model_weights:
        param.requires_grad = False
        if "_proj.weight" not in name and "W_pack.weight" not in name:
            continue
        for base_key, lora_keys in module_map.items():
            if not name.endswith(f"{base_key}.weight"):
                continue
            for stride_id, weight_key in enumerate(lora_keys):
                lora_a = name.replace(
                    f"{base_key}.weight", f"{weight_key}.lora_A.weight"
                )
                lora_b = name.replace(
                    f"{base_key}.weight", f"{weight_key}.lora_B.weight"
                )
                shard_size = param.shape[0] // len(lora_keys)
                if lora_a not in lora_weights:
                    continue
                assert lora_b in lora_weights, f"{lora_b} not in lora_weights"
                assert lora_weights[lora_b].shape[1] == r, (
                    f"{r=} != " f"{lora_weights[lora_b].shape}"
                )
                matrix = (
                    transpose(
                        lora_weights[lora_b] @ lora_weights[lora_a], fan_in_fan_out
                    )
                    * scaling
                )
                start_idx = shard_size * stride_id
                end_idx = shard_size * (stride_id + 1)
                assert param.data[start_idx:end_idx].shape == matrix.shape
                if merge:
                    param.data[start_idx:end_idx] += matrix
                else:
                    param.data[start_idx:end_idx] -= matrix
                switched_keys.add(lora_a)
                switched_keys.add(lora_b)
    no_replaced = [k for k in lora_weights.keys() if k not in switched_keys]
    assert len(no_replaced) == 0, (
        f"some lora states not loaded, " f"check again!: {no_replaced}"
    )
    base_model.is_merged = merge


def adapt_lora(llm, lora_weights, lora_config, merge=True):
    for worker in llm.llm_engine.workers:
        switch_lora(
            worker.model,
            lora_weights,
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            fan_in_fan_out=lora_config.fan_in_fan_out,
            merge=merge,
        )


class Lora_Config:
    def __init__(self, lora_config_path):
        lora_dict = json.load(open(lora_config_path))
        self.r = lora_dict["r"]
        self.lora_alpha = lora_dict["lora_alpha"]
        self.fan_in_fan_out = lora_dict["fan_in_fan_out"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-k",
        "--test_type",
        help="value: lora|merge ",
        default="merge",
    )
    args = parser.parse_args()
    model_path = "/home/tzw/models/baichuan2/baichuan2-7b-base_vllm"
    lora_merge_model_path = "/data2/tzw/adapters/baichuan2/2023-11-06-14-25_model"
    lora_merge_model_path = "/data2/tzw/adapters/baichuan2/2023-11-07-10-20_model"
    lora_parent = "/data2/tzw/adapters/baichuan2/2023-11-11-19-24"
    lora_parent = (
        "/home/tzw/models/adapters/baichuan2/baichuan2-7b-base/2023-11-12-14-40"
    )
    lora_model_path = f"{lora_parent}/adapter_model.bin"
    lora_config_path = f"{lora_parent}/adapter_config.json"
    inference_task_name = "generate"
    inference_config = (
        Path(__file__).parent
        / ".."
        / ".."
        / ".."
        / "prompts/generate/v1/inference_config.json"
    )
    inference_config = (
        "/home/tzw/dev/sft_hub/sfthub/classification/inference_config.json"
    )
    if args.test_type == "merge":
        baichun_llm = create_vllm(model_path=lora_merge_model_path, gpu_memory_need=16)
    else:
        baichun_llm = create_vllm(model_path=model_path, gpu_memory_need=16)
    import time

    begin = time.time()

    input = """文本：你想像自己的小花园
是这个香味的
那如果说你不喜欢有香味的话
你可以通过它这个
白色的是无香的
没有味道，请对其描述的产品进行输出，无产品或者无法判断的时候分别输出这两个标签，产品的一二级标签使用“-”进行连接
"""
    input = f"<reserved_106>{input}<reserved_107>"
    if args.test_type == "merge":
        response = baichun_llm.reason(
            input,
            inference_config_path=inference_config,
            inference_task_name=inference_task_name,
        )
        end_time = time.time()
        elapsed_time = end_time - begin
        elapsed_time_formatted = "{:.2f}".format(elapsed_time)
        print(f"input: {input}\nresponse:{response} {elapsed_time_formatted}\n")
    else:
        # merge lora
        logging.debug(f"【 lora switch start 】 {datetime.now()}")
        # TODO multi gpus adaptation
        lora_state = torch.load(lora_model_path, map_location="cuda:0")
        lora_config = Lora_Config(lora_config_path)
        adapt_lora(baichun_llm.model, lora_state, lora_config, merge=True)
        logging.debug(f"【 lora switch end 】 {datetime.now()}")

        # test here
        response = baichun_llm.reason(
            input,
            inference_config_path=inference_config,
            inference_task_name=inference_task_name,
        )
        end_time = time.time()
        elapsed_time = end_time - begin
        elapsed_time_formatted = "{:.2f}".format(elapsed_time)
        print(f"input: {input}\nresponse:{response} {elapsed_time_formatted}\n")

        # unmerge lora
        # Attention: the lora to unmerge should be the one merged
        logging.debug(f"【 lora switch start 】 {datetime.now()}")
        adapt_lora(baichun_llm.model, lora_state, lora_config, merge=False)
        logging.debug(f"【 lora switch end 】 {datetime.now()}")
