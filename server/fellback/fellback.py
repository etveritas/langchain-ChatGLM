import os
import logging
from typing import List
from fastapi import Body
from server.utils import BaseResponse

FELL_FORMAT = "%(asctime)s %(name)s %(message)s"
fell_logger = logging.getLogger("user_fellback")
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../logs")
handler = logging.FileHandler(filename=os.path.join(log_dir, "user_fellback.log"), encoding="utf-8")
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(FELL_FORMAT)
handler.setFormatter(formatter)
fell_logger.propagate = False

fell_logger.addHandler(handler)

def fellback_save(uuid: str = Body(..., description="该组对话唯一标识", examples=["2c24de88"]),
                  answer: str = Body(..., description="大模型回答内容", examples=["你好，有什么可以帮助你的吗？"]),
                  docs: List[str] = Body(..., description="相似度知识文档列表", examples=[["你好1", "你好2"]]),
                  prompt: str = Body(..., description="用户问题组装的Prompt", examples=["请根据一下内容回答问题：你好"])
                  ):
    # 日志文件路径
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../logs")
    try:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        fell_logger.info(f"""uuid: {uuid} answer: {str([answer])} docs: {str(docs)}, prompt: {str([prompt])}""")
        return BaseResponse(code=200, msg=f"用户反馈信息已接收")
    except:
        return BaseResponse(code=407, msg=f"用户反馈信息接收失败")
