from pydantic import BaseModel
import requests
import time
from easydict import EasyDict


class EventRequestMsg(BaseModel):
    request_id: str = "no request id"
    title: str = "美媒：美国猴痘疫情暴发不同寻常 数据、临床等问题堆积如山"
    content: str = """海外网9月15日电 据美国有线电视新闻网13日报道,美国的猴痘疫情暴发不同寻常,数据、临床护理和研究等相关问题堆积如山。
美国12日确认了首例人感染猴痘死亡病例,美国有线电视新闻网称,这是一个“悲惨的信号”,即尽管9月第一个星期,美国的猴痘病例数呈下降趋势,但疫情仍未平息,且对民众构成威胁。美国全国性病科主任联盟执行董事戴维·哈维在13日召开的记者会上表示,美国仍需加大力度应对猴痘疫情。“从近几十年来看,这种病毒在美国的暴发非同寻常,数据、临床护理和研究等相关问题堆积如山。”
哈维及其他美国公共卫生部门负责人还警告称,处于医疗第一线的工作人员(包括传染病专家、当地卫生部门和诊所等人员)没有足够的资源来确保情况改善。一些关键医疗领域存在疫苗接种不公平、监测数据不完整等问题。
美国疾病预防控制中心此前数据显示,该国猴痘感染人群中存在明显的种族差异,有色人种病例占比高,但疫苗接种比例很低。在过去两个月中,仅占美国总人口约30%的拉美裔和非洲裔在美国猴痘确诊病例中占比超过60%,然而仅有10%的猴痘疫苗被分配给了占全美确诊病例约33%的非裔病例接种。(海外网/王珊宁)
【编辑:周驰】
更多精彩内容请进入国际频道"""
    pub_time: str = """15/9/2022 15:53:05"""


def call_event_extraction_proxy(url, event_msg_request: EventRequestMsg):
    whole_begin = time.time()
    response_dict = EasyDict()
    response_dict.code = 500
    try:
        headers = {"accept": "application/json", "Content-Type": "application/json"}
        response = requests.post(
            url, headers=headers, json=event_msg_request.model_dump()
        )
        if response.status_code != 200:
            response_dict.message = "something wrong"
            response_dict.code = response.status_code
        else:
            return response.json()
    except Exception as e:
        response_dict.code = 500
        response_dict.message = str(e)
    end_time = time.time()
    elapsed_time = end_time - whole_begin
    elapsed_time_formatted = "{:.2f}".format(elapsed_time)
    response_dict.cost = elapsed_time_formatted
    response_dict.data = EasyDict()
    response_dict.data.trace_id = event_msg_request.request_id
    response_dict.data.extraction_result = []
    return response_dict


def main():
    url = "http://localhost:8771/infectious_event_extraction"
    request_data = EventRequestMsg()
    data = call_event_extraction_proxy(url, request_data)
    print(type(data), data)
    data = call_event_extraction_proxy(url + "1", request_data)
    print(type(data), data)


if __name__ == "__main__":
    main()
