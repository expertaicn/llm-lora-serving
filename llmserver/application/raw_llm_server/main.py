from vllm import LLM, SamplingParams

if __name__ == "__main__":
    model_name = "/home/tzw/models/baichuan2/baichuan2-7b-base"
    llm = LLM(model=model_name, trust_remote_code=True)
    query = "中国的首都是"
    input_pattern = "{}"
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=256)
    outputs = llm.generate([input_pattern.format(query)], sampling_params)
    for output in outputs:
        print(output)
        generated_text = output.outputs[0].text
        generated_text = (
            generated_text.strip().replace("</s>", "").replace("<s>", "").strip()
        )
        print(f"query: {query}\nanswer: {generated_text}")
