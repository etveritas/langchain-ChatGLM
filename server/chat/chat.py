from fastapi import Body
from fastapi.responses import StreamingResponse
from configs.model_config import llm_model_dict, LLM_MODEL
from configs.model_config import QTPL_PROMPT, KTPL_PROMPT
from server.chat.utils import wrap_done
from models.chatglm import ChatChatGLM
from langchain.llms import ChatGLM, OpenAI
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable
import asyncio
from langchain.prompts.chat import ChatPromptTemplate
from typing import List
from server.chat.utils import History
import json
import uuid


def chat(query: str = Body(..., description="用户输入", examples=["恼羞成怒"]),
         history: List[History] = Body([],
                                       description="历史对话",
                                       examples=[[
                                           {"role": "user", "content": "我们来玩成语接龙，我先来，生龙活虎"},
                                           {"role": "assistant", "content": "虎头虎脑"}]]
                                       ),
         stream: bool = Body(False, description="流式输出"),
         ):
    history = [History(**h) if isinstance(h, dict) else h for h in history]

    async def chat_iterator(query: str,
                            history: List[History] = [],
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

        chat_prompt = ChatPromptTemplate.from_messages(
            [i.to_msg_tuple() for i in history]
            + [("human", KTPL_PROMPT)]
            + [("human", QTPL_PROMPT)]
        )
        chain = LLMChain(prompt=chat_prompt, llm=model)

        # combine prompt
        prompt_comb = chain.prep_prompts([{"context": query, "question": query}])
        # Begin a task that runs in the background.
        task = asyncio.create_task(wrap_done(
            chain.acall({"context": query, "question": query}),
            callback.done),
        )

        unq_id = uuid.uuid1()
        if stream:
            async for token in callback.aiter():
                # Use server-sent-events to stream the response
                yield json.dumps({"uuid": str(unq_id),
                                  "answer": token,
                                  "docs": [],
                                  "reference": {},
                                  "prompt": prompt_comb[0][0].to_string()},
                                  ensure_ascii=False)
        else:
            answer = ""
            async for token in callback.aiter():
                answer += token
            yield json.dumps({"uuid": str(unq_id),
                              "answer": answer,
                              "docs": [],
                              "reference": {},
                              "prompt": prompt_comb[0][0].to_string()},
                              ensure_ascii=False)

        await task

    return StreamingResponse(chat_iterator(query, history),
                             media_type="text/event-stream")
