import uvicorn
import time
from easydict import EasyDict
from fastapi import FastAPI
from pydantic import BaseModel
import argparse
from pathlib import Path
import os
import logging
import sys

# from llmserver.config import Config
import traceback
from llmserver.log import config_log
from vllm import LLM, SamplingParams
from fastapi import FastAPI
from typing import List

# from fastapi.middleware.cors import CORSMiddleware

import threading

lock = threading.Lock()

app = FastAPI()
config_log(
    project="llmserver",
    module="batchserver",
    print_termninal=True,
    level=logging.INFO,
)
model_name = "/home/tzw/models/baichuan2/baichuan2_13b_vllm/"
llm = LLM(model=model_name, trust_remote_code=True, tensor_parallel_size=2)
query = "中国的首都是"
input_pattern = "{}"
sampling_params = SamplingParams(temperature=0.1, top_p=0.95, max_tokens=256)


class RequestMsg(BaseModel):
    requestID: str = "no request id"
    prompt_str_list: List[str] = ["美国的首都是？", "中国的人口有多少？"]
    inference_config_path: Path = Path("/tmp/tzw.txt")
    inference_task_name: str = "generate"
    stop_list = ([],)
    lora_name = (None,)


@app.post("/text_generation")
def batch_echo(request_msg: RequestMsg):
    response = EasyDict()
    whole_begin = time.time()
    results = []
    outputs = llm.generate(request_msg.prompt_str_list, sampling_params)
    for output in outputs:
        generated_text = output.outputs[0].text
        results.append(generated_text)
        logging.debug(f"output: [{generated_text}]")

        end_time = time.time()
    elapsed_time = end_time - whole_begin
    elapsed_time_formatted = "{:.2f}".format(elapsed_time)
    response.code = 200
    response.message = "OK!"
    response.data = EasyDict()
    response.data.requestID = request_msg.requestID
    response.data.list = results
    response.cost = elapsed_time_formatted
    return response


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", help="env:prod|test", default="prod")
    parser.add_argument(
        "-c",
        "--config",
        help="config: config file ",
        default="config.yml",
    )
    parser.add_argument(
        "-g",
        "--gpu_index",
        help="cuda number",
        default=None,
    )
    args = parser.parse_args()
    root_dir = Path(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    )
    # init config
    # config_file = root_dir / "etc" / args.env / args.config
    # if not config_file.is_file():
    #     sys.stderr.write(f"{config_file} does not exist")
    #     sys.exit(2)
    # with open(config_file, "r") as file:
    #     config = Config.load(config_file)
    # cuda_index = args.gpu_index
    # if cuda_index is not None:
    #     cuda_index = int(cuda_index)
    # init service
    # SimpleGenerateEngine.instance().build(config, cuda_device=cuda_index)
    uvicorn.run(app, host="0.0.0.0", port=8172)
