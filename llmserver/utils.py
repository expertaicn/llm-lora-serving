import logging
import random
import time
from collections import OrderedDict
from typing import Optional, Tuple, List
import inspect
import GPUtil
import yaml


def ordered_yaml_load(yaml_path, Loader=yaml.FullLoader, object_pairs_hook=OrderedDict):
    class OrderedLoader(Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))

    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping
    )
    with open(yaml_path) as stream:
        return yaml.load(stream, OrderedLoader)


def ordered_yaml_dump(data, stream=None, Dumper=yaml.SafeDumper, **kwds):
    class OrderedDumper(Dumper):
        pass

    def _dict_representer(dumper, data):
        return dumper.represent_mapping(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, data.items()
        )

    OrderedDumper.add_representer(OrderedDict, _dict_representer)
    return yaml.dump(
        data,
        stream,
        OrderedDumper,
        default_flow_style=False,
        encoding="utf-8",
        allow_unicode=True,
        **kwds,
    )


def get_available_gpu_with_free_memory(
    min_free_memory_gb: int,
    default_gpu_index: Optional[int] = None,
    exclude_list: List[int] = [],
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """get available gpu with free memory

    Args:
        min_free_memory_gb (int): GPU memory size, in gigabytes (GB)
        default_gpu_index (int, optional): default gput index. Defaults to None.
        exclude_set (tuple, optional): gpu index list. Defaults to (0, 1, 2, 3).

    Returns:
        Optional[int]: none, none, none;  or gpu_index, total_memory, memory_free
    """

    try:
        gpus = GPUtil.getGPUs()
        available_gpus_with_free_memory = []

        for gpu_index, gpu in enumerate(gpus):
            if gpu_index in exclude_list:
                continue
            memory_total = gpu.memoryTotal
            memroy_free = gpu.memoryFree
            if memroy_free > min_free_memory_gb * 1024:  # 转换为MB
                available_gpus_with_free_memory.append(
                    (gpu_index, memory_total, memroy_free)
                )
        if default_gpu_index is not None:
            default_gpu_index = int(default_gpu_index)
            for item in available_gpus_with_free_memory:
                if default_gpu_index == item[0]:
                    return item[0], item[1], item[2]
            return None, None, None
        if available_gpus_with_free_memory:
            selected_gpu = random.choice(available_gpus_with_free_memory)
            return selected_gpu[0], selected_gpu[1], selected_gpu[2]
        else:
            logging.error(
                f"No GPU with more than {min_free_memory_gb}GB free memory available."
            )
    except Exception as e:
        logging.error("try to get available gup : Error occurred:", e)
    return None, None, None


def cost_time_from(begin):
    end_time = time.time()
    elapsed_time = end_time - begin
    elapsed_time_formatted = "{:.2f}".format(elapsed_time)
    return elapsed_time_formatted


def timeit(func):
    """装饰器，度量函数的执行时间并使用logging打印"""

    def wrapper(*args, **kwargs):
        start_time = time.time()  # 获取开始时间
        file_name = inspect.getfile(func)
        file_name = file_name.split("/")[-1]
        result = func(*args, **kwargs)  # 调用原始函数
        end_time = time.time()  # 获取结束时间
        elapsed_time = end_time - start_time  # 计算函数执行时间
        logging.info(
            f"{file_name}:{func.__name__} took {elapsed_time:.4f} seconds to execute"
        )
        return result

    return wrapper


if __name__ == "__main__":
    yml_path = "/tmp/order.dump.yml"
    out = open(yml_path, "w")
    a = {
        "z": "he",
        "d": "hhh",
        "c": ["上", "中", "下"],
        "b": {"a": "xxx", "z": "xxx", "d": "xxx"},
        "e": [["a", "b", "c"], ["e", "f", "g"]],
    }

    a = {
        "constraints": [
            "~4000 word limit for short term memory. Your short term memory is short, so immediately save important information to  files.",
            "If you are unsure how you previously did something or want to recall past events, thinking about similar events will   help you remember.",
            "No user assistance",
            "Exclusively use the commands listed below e.g. command_name",
        ],
        "resources": [
            "Internet access for searches and information gathering.",
            "Long Term memory management.",
            "GPT-3.5 powered Agents for delegation of simple tasks.",
            "File output.",
        ],
        "performance_evaluations": [
            "Continuously review and analyze your actions to ensure you are performing to the best of your abilities.",
            "Constructively self-criticize your big-picture behavior constantly.",
            "Reflect on past decisions and strategies to refine your approach.",
            "Every command has a cost, so be smart and efficient. Aim to complete tasks in the least number of steps.",
            "Write all code to a file.",
        ],
    }
    ordered_yaml_dump(a, out)
    b = ordered_yaml_load(yml_path)
    print(b)
    # 指定所需的最小空闲显存大小（GB）
    for min_free_memory_gb in [16, 30]:
        selected_gpu = get_available_gpu_with_free_memory(
            min_free_memory_gb, default_gpu_index=None
        )

        if selected_gpu is not None:
            print(
                f"Using GPU {selected_gpu} with more than {min_free_memory_gb}GB free memory."
            )
        else:
            print(
                f"No GPU {selected_gpu} with more than {min_free_memory_gb}GB free memory."
            )
