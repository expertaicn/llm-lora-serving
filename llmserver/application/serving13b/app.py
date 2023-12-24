from vllm import LLM, SamplingParams

# export CUDA_VISIBLE_DEVICES=1,2 && python app.py
if __name__ == "__main__":
    model_name = "/home/tzw/models/baichuan2/baichuan2_13b_vllm/"
    # model_name = "/home/tzw/models/tmp/shibing624/ziya-llama-13b-medical-merged"
    model_name = "/home/tzw/models/01-ai/Yi-34B-Chat-4bits/"
    # model_name = "/home/tzw/models/01-ai/Yi-34B-Chat-8bits/"
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        tensor_parallel_size=4,
        quantization="awq",
    )
    query = "一岁宝宝发烧能吃啥药"
    query = "肛门病变可能是什么疾病的症状"
    input_pattern = """Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:{}\n\n### Response: """
    input_pattern = """<|im_start|>system\n回答问题的时候请将字数控制在100字以内。谢谢。<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"""
    # input_pattern = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:{}\n\n### Response:"
    sampling_params = SamplingParams(
        temperature=0.1, top_p=0.85, max_tokens=1024, frequency_penalty=1.05
    )
    outputs = llm.generate([input_pattern.format(query)], sampling_params)
    for output in outputs:
        print(output)
        generated_text = output.outputs[0].text
        generated_text = (
            generated_text.strip().replace("</s>", "").replace("<s>", "").strip()
        )
        print(f"query: {query}\nanswer: {generated_text}")
