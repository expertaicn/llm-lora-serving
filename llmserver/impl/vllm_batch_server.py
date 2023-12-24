from vllm import LLM, SamplingParams
from abc import ABCMeta
from pathlib import Path
import json
import logging
from typing import Optional, List
from llmserver.custom_types import SingletonType
from llmserver.base_llm_server import BaseLLMServer
from llmserver.config import ServerConfig, LoraInfo, InferenceParam
import threading
import copy
import torch
from datetime import datetime
from llmserver.config import LoraInfo


##############################################################
#
# LORA Adapter TODO 分布式环境的lora合并存在问题，主要的原因是不太理解ray的原理
#
##############################################################


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
        try:
            model = worker.model
        except:
            return
            # print(dir(worker), type(worker))
            # try:
            #     tmp = worker.execute_method.remote
            #     print(type(tmp))
            #     tmp = worker.__getattr__.remote

            #     print(type(tmp))
            #     model = tmp("model")
            #     print(type(model))
            #     import ray

            #     model = ray.get(model)
            #     print(type(model))
            #     # tmp = worker.getattr("model")
            #     # print(type(tmp))
            # except:
            #     import traceback

            #     print(traceback.format_exc())
            #     print("worker has no member worker")

        switch_lora(
            model,
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


def lora_decorator(func):
    def wrapper(self, *args, **kwargs):
        with self.lock:
            # 提取 merge 和 name 参数
            lora_name = kwargs.get("lora_name", None)
            lora_name = None  # IMPROVE ME
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


class VllmBatchServer(BaseLLMServer, metaclass=VLLMMetaClass):
    @classmethod
    def instance(cls):
        return VllmBatchServer()

    def __init__(self) -> None:
        super().__init__()
        self.lock = threading.Lock()  # 创建一个线程锁
        self.default_param = SamplingParams(temperature=0.1, top_p=0.95, max_tokens=256)

    def build(self, server_config: ServerConfig):
        super().build(server_config=server_config)
        # 这里我们不自动找gpu
        self.model = LLM(
            model=server_config.base_model_name_or_path,
            trust_remote_code=True,
            tensor_parallel_size=server_config.tensor_parallelism_size,
            # quantization="awq",
            # dtype="float16",  # TODO when use lora, should uncomment this line
        )

    def reason(
        self,
        prompt_str: str,
        inference_type_name: str,
        inference_param: InferenceParam,
        lora_name: str,
    ) -> Optional[str]:
        results = self.batch_reason(
            [prompt_str],
            inference_type_name=inference_type_name,
            inference_pram=inference_param,
            lora_name=lora_name,
        )
        if len(results) == 0:
            return None
        else:
            return results[0]

    def _get_inference_param(
        self, inference_type_name: str, inference_param: InferenceParam, n=1
    ):
        vllm_inference_param = self.default_param

        if inference_param is None:
            inference_param = self.inference_type_dict.get(inference_type_name, None)
        if inference_param is not None:
            temperature = inference_param.temperature
            top_p = inference_param.top_p
            top_k = inference_param.top_k
            max_tokens = inference_param.max_tokens
            stop_list = inference_param.stop_list
            vllm_inference_param = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                stop=stop_list,
                n=n,
            )
        return vllm_inference_param

    @lora_decorator
    def batch_reason(
        self,
        prompt_str_list: List[str],
        inference_type_name: str,
        inference_param: InferenceParam = None,
        lora_name=None,
    ) -> List[str]:
        vllm_inference_param = self._get_inference_param(
            inference_param=inference_param, inference_type_name=inference_type_name
        )
        results = []
        outputs = self.model.generate(prompt_str_list, vllm_inference_param)
        for output in outputs:
            generated_text = output.outputs[0].text
            results.append(generated_text)
            logging.debug(f"output: [{generated_text}]")
        return results

    @lora_decorator
    def multi_reason(
        self,
        prompt_str_list: List[str],
        inference_type_name: str,
        inference_param: InferenceParam,
        n: int = 0,
        stop_list=[],
        lora_name=None,
    ) -> List[List[str]]:
        vllm_inference_param = self._get_inference_param(
            inference_param=inference_param, inference_type_name=inference_type_name
        )
        batch_results = []
        batch_outputs = self.model.generate(prompt_str_list, vllm_inference_param)
        for outputs in batch_outputs:
            results = []
            for output in outputs.outputs:
                generated_text = output.text
                results.append(generated_text)
                logging.debug(f"output: [{generated_text}]")
            batch_results.append(results)
        return batch_results


if __name__ == "__main__":
    pass
    file_path = (
        Path(__file__).parent / ".." / ".." / "etc" / "dev" / "server_13b_sft.yml"
    )
    config = ServerConfig.load(file_path=file_path)
    print(config.base_model_name_or_path)
    print(config.lora_list)
    vllm_batch_server = VllmBatchServer()
    vllm_batch_server.build(server_config=config)
    print(vllm_batch_server.lora_dict)
    text = '''"文本：你想像自己的小花园\n是这个香味的\n那如果说你不喜欢有香味的话\n你可以通过它这个\n白色的是无香的\n没有味道，
请对其描述的产品进行输出，无产品或者无法判断的时候分别输出这两个标签，产品的一二级标签使用“-”进行连接"'''

    input_list = [text]
    prefix = "<reserved_106>"
    postfix = "<reserved_107>"
    # prefix = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nHuman: "
    # postfix = "\nAssistant:"
    input_list = [f"{prefix}{render_str}{postfix}" for render_str in input_list]
    result = vllm_batch_server.batch_reason(
        prompt_str_list=input_list,
        inference_type_name="baichuan2_strong_certainty",
    )
    print(result)
