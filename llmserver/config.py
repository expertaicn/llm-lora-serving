import logging
from pydantic import BaseModel
from llmserver.utils import ordered_yaml_load
from typing import List
from pathlib import Path


class LoraInfo(BaseModel):
    lora_name: str
    lora_parent: str
    lora_model_path: str = "adapter_model.bin"
    lora_config_path: str = "adapter_config.json"
