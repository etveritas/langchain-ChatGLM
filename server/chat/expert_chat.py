from fastapi import Body, Request
from fastapi.responses import StreamingResponse
from configs.model_config import (llm_model_dict, LLM_MODEL, PROMPT_TEMPLATE,
                                  VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD)
from configs.model_config import (QTPL_PROMPT, KTPL_PROMPT, IT_PROMPT, RISK_PROMPT, HRES_PROMPT,
                                  OP_PROMPT, MNY_PROMPT, FIN_PROMPT, MKT_PROMPT)
from server.chat.utils import wrap_done
from server.utils import BaseResponse
from langchain.chat_models import ChatOpenAI
from models.chatglm import ChatChatGLM
from langchain.llms import ChatGLM, OpenAI
from langchain import LLMChain
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable, List, Optional
import asyncio
from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from server.chat.utils import History
from server.knowledge_base.kb_service.base import KBService, KBServiceFactory
import json
import os
import uuid
import numpy as np
from urllib.parse import urlencode
from server.knowledge_base.kb_doc_api import search_docs
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain


def sigmoid(x):
    return 1/(1+np.exp(-x))

def expert_chat(query: str = Body(..., description="用户输入", examples=["你好"]),
                        knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
                        top_k: int = Body(VECTOR_SEARCH_TOP_K, description="匹配向量数"),
                        score_threshold: float = Body(SCORE_THRESHOLD, description="知识库匹配相关度阈值，取值范围在0-1之间，SCORE越小，相关度越高，取到1相当于不筛选，建议设置在0.5左右", ge=0, le=1100),
                        history: List[History] = Body([],
                                                      description="历史对话",
                                                      examples=[[
                                                          {"role": "user",
                                                           "content": "我们来玩成语接龙，我先来，生龙活虎"},
                                                          {"role": "assistant",
                                                           "content": "虎头虎脑"}]]
                                                      ),
                        stream: bool = Body(False, description="流式输出"),
                        local_doc_url: bool = Body(False, description="知识文件返回本地路径(true)或URL(false)"),
                        request: Request = None,
                        sys_status: str = Body(..., description="流程节点", examples=[""])
                        ):
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    history = [History(**h) if isinstance(h, dict) else h for h in history]

    async def expert_chat_iterator(query: str,
                                           kb: KBService,
                                           top_k: int,
                                           history: Optional[List[History]],
                                           ) -> AsyncIterable[str]:
        callback = AsyncIteratorCallbackHandler()
        if "gpt" in LLM_MODEL:
            model = ChatOpenAI(
                top_p=0.9,
                temperature=0.6,
                streaming=True,
                verbose=True,
                callbacks=[callback],
                openai_api_key=llm_model_dict[LLM_MODEL]["api_key"],
                openai_api_base=llm_model_dict[LLM_MODEL]["api_base_url"],
                model_name=LLM_MODEL
            )
        elif "glm" in LLM_MODEL:
            model = ChatChatGLM(
                top_p=0.9,
                temperature=0.6,
                streaming=True,
                verbose=True,
                callbacks=[callback],
                chatglm_api_key=llm_model_dict[LLM_MODEL]["api_key"],
                chatglm_api_base=llm_model_dict[LLM_MODEL]["api_base_url"],
                model_name=LLM_MODEL
            )

        # output_parser = CommaSeparatedListOutputParser()
        # chatp = PromptTemplate(template=RECOGNIZE_PROMPT, input_variables=["user_query"], output_parser=output_parser)
        # # recg_prompt = chatp.from_messages([("human", RECOGNIZE_PROMPT)])
        # chain_recg = LLMChain(prompt=chatp, llm=model)
        # qry_kw = chain_recg.predict(user_query=query)
        # print("qry_kw:", qry_kw)
        if knowledge_base_name == "caxins_it":
            chat_prompt = ChatPromptTemplate.from_messages(
                [("human", IT_PROMPT)]
                + [i.to_msg_tuple() for i in history]
                + [("human", QTPL_PROMPT)]
                + [("human", KTPL_PROMPT)]
            )
            chain = LLMChain(prompt=chat_prompt, llm=model)
            if sys_status == "":
                # combine prompt
                pmt_comb1 = chain.prep_prompts([{"context": "第一步", "question": query}])
                print(pmt_comb1[0][0].dict())
                print("pmt_comb1:\n", pmt_comb1[0][0].to_string())
                asts_res1 = chain.run({"context": "第一步", "question": query})
                yield asts_res1
            elif "完成用户意图理解" in sys_status:
                pmt_comb2 = chain.prep_prompts([{"context": "第二步", "question": "请判断是否需要查询知识库。"}])
                print("pmt_comb2:\n", pmt_comb2[0][0].to_string())
                asts_res2 = chain.run({"context": "第二步", "question": "请判断是否需要查询知识库。"})
                print("asts_res2", asts_res2)
            elif "知识库需要查询" in sys_status:
                pmt_comb3 = chain.prep_prompts([{"context": "第三步", "question": "请教最近历史对话提炼成一句话。"}])
                print("pmt_comb3:\n", pmt_comb3[0][0].to_string())
                asts_res3 = chain.run({"context": "第三步", "question": "请教最近历史对话提炼成一句话。"})
                print("asts_res3", asts_res3)
                docs = search_docs(asts_res3, knowledge_base_name, top_k, score_threshold)
                print(docs)
                context = "\n".join([os.path.split(doc.metadata["source"])[-1]+":"+doc.page_content for doc in docs])
                pmt_comb4 = chain.prep_prompts([{"context": context, "question": asts_res3}])
                print("context", context)
                print("pmt_comb4:\n", pmt_comb4[0][0].to_string())
                asts_res4 = chain.run({"context": context, "question": asts_res3})
                print("asts_res4", asts_res4)
            else:
                pmt_comb5 = chain.prep_prompts([{"context": "第二步", "question": asts_res2}])
                print("pmt_comb5:\n", pmt_comb5[0][0].to_string())
                asts_res5 = chain.run({"context": "第二步", "question": asts_res2})
                print("asts_res5", asts_res5)

            # if stream:
            #     async for token in callback.aiter():
            #         # Use server-sent-events to stream the response
            #         yield json.dumps({"uuid": str(unq_id),
            #                         "answer": token,
            #                         "docs": source_documents,
            #                         "prompt": prompt_comb[0][0].to_string()},
            #                         ensure_ascii=False)
            # else:
            #     answer = ""
            #     async for token in callback.aiter():
            #         answer += token
            #     yield json.dumps({"uuid": str(unq_id),
            #                     "answer": answer,
            #                     "docs": source_documents,
            #                     "prompt": prompt_comb[0][0].to_string()},
            #                     ensure_ascii=False)
            # await task
        elif knowledge_base_name == "caxins_risk":
            chat_prompt = ChatPromptTemplate.from_messages(
                [("human", RISK_PROMPT)]
                + [i.to_msg_tuple() for i in history]
                + [("human", QTPL_PROMPT)]
                + [("human", KTPL_PROMPT)]
            )
        elif knowledge_base_name == "caxins_hres":
            chat_prompt = ChatPromptTemplate.from_messages(
                [("human", HRES_PROMPT)]
                + [i.to_msg_tuple() for i in history]
                + [("human", QTPL_PROMPT)]
                + [("human", KTPL_PROMPT)]
            )
        elif knowledge_base_name == "caxins_op":
            chat_prompt = ChatPromptTemplate.from_messages(
                [("human", OP_PROMPT)]
                + [i.to_msg_tuple() for i in history]
                + [("human", QTPL_PROMPT)]
                + [("human", KTPL_PROMPT)]
            )
        elif knowledge_base_name == "caxins_mny":
            chat_prompt = ChatPromptTemplate.from_messages(
                [("human", MNY_PROMPT)]
                + [i.to_msg_tuple() for i in history]
                + [("human", QTPL_PROMPT)]
                + [("human", KTPL_PROMPT)]
            )
        elif knowledge_base_name == "caxins_fin":
            chat_prompt = ChatPromptTemplate.from_messages(
                [("human", FIN_PROMPT)]
                + [i.to_msg_tuple() for i in history]
                + [("human", QTPL_PROMPT)]
                + [("human", KTPL_PROMPT)]
            )
        elif knowledge_base_name == "caxins_mkt":
            chat_prompt = ChatPromptTemplate.from_messages(
                [("human", MKT_PROMPT)]
                + [i.to_msg_tuple() for i in history]
                + [("human", QTPL_PROMPT)]
                + [("human", KTPL_PROMPT)]
            )
        else:
            chat_prompt = ChatPromptTemplate.from_messages(
                [i.to_msg_tuple() for i in history]
                + [("human", PROMPT_TEMPLATE)]
            )


        # docs = search_docs(query, knowledge_base_name, top_k, score_threshold)
        # context = "\n".join([doc.page_content for doc in docs])

        # chat_prompt = ChatPromptTemplate.from_messages(
        #     [i.to_msg_tuple() for i in history] + [("human", PROMPT_TEMPLATE)])

        # chain = LLMChain(prompt=chat_prompt, llm=model)

        # # combine prompt
        # prompt_comb = chain.prep_prompts([{"context": context, "question": query}])

        # # Begin a task that runs in the background.
        # task = asyncio.create_task(wrap_done(
        #     chain.acall({"context": context, "question": query}),
        #     callback.done),
        # )

        # source_documents = []
        # for inum, doc in enumerate(docs):
        #     filename = os.path.split(doc.metadata["source"])[-1]
        #     if local_doc_url:
        #         url = "file://" + doc.metadata["source"]
        #     else:
        #         parameters = urlencode({"knowledge_base_name": knowledge_base_name, "file_name":filename})
        #         url = f"{request.base_url}knowledge_base/download_doc?" + parameters
        #     text = f"""出处 [{inum + 1}] [{filename}]({url}) \n\n{doc.page_content}\n\n 相似度：{1100-doc.score}\n\n"""
        #     source_documents.append(text)

        # unq_id = uuid.uuid1()
        # if stream:
        #     async for token in callback.aiter():
        #         # Use server-sent-events to stream the response
        #         yield json.dumps({"uuid": str(unq_id),
        #                           "answer": token,
        #                           "docs": source_documents,
        #                           "prompt": prompt_comb[0][0].to_string()},
        #                          ensure_ascii=False)
        # else:
        #     answer = ""
        #     async for token in callback.aiter():
        #         answer += token
        #     yield json.dumps({"uuid": str(unq_id),
        #                       "answer": answer,
        #                       "docs": source_documents,
        #                       "prompt": prompt_comb[0][0].to_string()},
        #                      ensure_ascii=False)

        # await task

    return StreamingResponse(expert_chat_iterator(query, kb, top_k, history),
                             media_type="text/event-stream")
