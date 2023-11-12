import streamlit as st
from vllm import LLM, SamplingParams
import time
import streamlit.components.v1 as components
import os

# 设置要使用的GPU编号，例如使用GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
input_pattern = "{}"
st.write("# 百川13B #! 👋")

selected_option = st.sidebar.radio("选择一个选项", ["原生", "Firely-Finetune"], index=1)
if selected_option == "原生":
    input_pattern = "{}"
if selected_option == "Firely-Finetune":
    input_pattern = "<s>{}</s>"


@st.cache_resource
def get_llm_service():
    model_name = "/run/tzw/checkpoint/firefly-baichuan-13b-qlora-sft-merge"
    model_name = "/run/tzw/code_lllama"
    model_name = "/run/tzw/models/baichuan2/baichuan2"
    model_name = "/run/tzw/models/baichuan2/baichuan2_vllm"
    model_name = "/home/tzw/models/baichuan2/baichuan2-7b-base"
    llm = LLM(model=model_name, trust_remote_code=True)
    return llm


if __name__ == "__main__":
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=256)

    query = st.text_area("问题：")
    search_button = st.button("ASK: LLM")
    llm = get_llm_service()

    if search_button:
        begin_time = time.time()
        outputs = llm.generate([input_pattern.format(query)], sampling_params)
        for output in outputs:
            cost_time = time.time() - begin_time
            st.write(f"cost: {cost_time:.4f}\n\nAnswers:\n\n")
            print(output)
            generated_text = output.outputs[0].text
            generated_text = (
                generated_text.strip().replace("</s>", "").replace("<s>", "").strip()
            )
            components.html(generated_text)
