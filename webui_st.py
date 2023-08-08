import streamlit as st
from streamlit_chatbox import st_chatbox
import tempfile
from pathlib import Path

###### 从webui借用的代码 #####
######   做了少量修改    #####
import os
import shutil

from chains.local_doc_qa import LocalDocQA
from configs.model_config import *
import nltk
from models.base import (BaseAnswer,
                         AnswerResult,)
import models.shared as shared
from models.loader.args import parser
from models.loader import LoaderCheckPoint

nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path


def get_vs_list():
    lst_default = ["新建知识库"]
    if not os.path.exists(KB_ROOT_PATH):
        return lst_default
    lst = os.listdir(KB_ROOT_PATH)
    lst = [x for x in lst if os.path.isdir(os.path.join(KB_ROOT_PATH, x))]
    if not lst:
        return lst_default
    lst.sort()
    return lst_default + lst


embedding_model_dict_list = list(embedding_model_dict.keys())
llm_model_dict_list = list(llm_model_dict.keys())


def get_answer(query, vs_path, history, mode, score_threshold=VECTOR_SEARCH_SCORE_THRESHOLD,
               vector_search_top_k=VECTOR_SEARCH_TOP_K, chunk_conent: bool = True,
               chunk_size=CHUNK_SIZE, streaming: bool = STREAMING,):
    if mode == "Bing搜索问答":
        for resp, history in local_doc_qa.get_search_result_based_answer(
                query=query, chat_history=history, streaming=streaming):
            source = "\n\n"
            source += "".join(
                [f"""<details> <summary>出处 [{i + 1}] <a href="{doc.metadata["source"]}" target="_blank">{doc.metadata["source"]}</a> </summary>\n"""
                 f"""{doc.page_content}\n"""
                 f"""</details>"""
                 for i, doc in
                 enumerate(resp["source_documents"])])
            history[-1][-1] += source
            yield history, ""
    elif mode == "知识库问答" and vs_path is not None and os.path.exists(vs_path):
        local_doc_qa.top_k = vector_search_top_k
        local_doc_qa.chunk_conent = chunk_conent
        local_doc_qa.chunk_size = chunk_size
        for resp, history in local_doc_qa.get_knowledge_based_answer(
                query=query, vs_path=vs_path, chat_history=history, streaming=streaming):
            source = "\n\n"
            source += "".join(
                [f"""<details> <summary>出处 [{i + 1}] {os.path.split(doc.metadata["source"])[-1]}</summary>\n"""
                 f"""{doc.page_content}\n"""
                 f"""</details>"""
                 for i, doc in
                 enumerate(resp["source_documents"])])
            history[-1][-1] += source
            yield history, ""
    elif mode == "知识库测试":
        if os.path.exists(vs_path):
            resp, prompt = local_doc_qa.get_knowledge_based_conent_test(query=query, vs_path=vs_path,
                                                                        score_threshold=score_threshold,
                                                                        vector_search_top_k=vector_search_top_k,
                                                                        chunk_conent=chunk_conent,
                                                                        chunk_size=chunk_size)
            if not resp["source_documents"]:
                yield history + [[query,
                                  "根据您的设定，没有匹配到任何内容，请确认您设置的知识相关度 Score 阈值是否过小或其他参数是否正确。"]], ""
            else:
                source = "\n".join(
                    [
                        f"""<details open> <summary>【知识相关度 Score】：{doc.metadata["score"]} - 【出处{i + 1}】：  {os.path.split(doc.metadata["source"])[-1]} </summary>\n"""
                        f"""{doc.page_content}\n"""
                        f"""</details>"""
                        for i, doc in
                        enumerate(resp["source_documents"])])
                history.append([query, "以下内容为知识库中满足设置条件的匹配结果：\n\n" + source])
                yield history, ""
        else:
            yield history + [[query,
                              "请选择知识库后进行测试，当前未选择知识库。"]], ""
    else:
        answer_result_stream_result = local_doc_qa.llm_model_chain(
            {"prompt": query, "history": history, "streaming": streaming})

        for answer_result in answer_result_stream_result['answer_result_stream']:
            resp = answer_result.llm_output["answer"]
            history = answer_result.history
            history[-1][-1] = resp + (
                "\n\n当前知识库为空，如需基于知识库进行问答，请先加载知识库后，再进行提问。" if mode == "知识库问答" else "")
            yield history, ""
    logger.info(f"flagging: username={FLAG_USER_NAME},query={query},vs_path={vs_path},mode={mode},history={history}")


def get_vector_store(vs_id, files, sentence_size, history, one_conent, one_content_segmentation):
    vs_path = Path(KB_ROOT_PATH) / vs_id / "vector_store"
    con_path = Path(KB_ROOT_PATH) / vs_id / "content"
    con_path.mkdir(parents=True, exist_ok=True)

    qa = st.session_state.local_doc_qa
    if qa.llm_model_chain and qa.embeddings:
        filelist = []
        if isinstance(files, list):
            for file in files:
                filename = os.path.split(file.name)[-1]
                target = con_path / filename
                shutil.move(file.name, target)
                filelist.append(str(target))
            vs_path, loaded_files = qa.init_knowledge_vector_store(
                filelist, str(vs_path), sentence_size)
        else:
            vs_path, loaded_files = qa.one_knowledge_add(str(vs_path), files, one_conent, one_content_segmentation,
                                                                   sentence_size)
        if len(loaded_files):
            file_status = f"已添加 {'、'.join([os.path.split(i)[-1] for i in loaded_files if i])} 内容至知识库，并已加载知识库，请开始提问"
        else:
            file_status = "文件未成功加载，请重新上传文件"
    else:
        file_status = "模型未完成加载，请先在加载模型后再导入文件"
        vs_path = None
    logger.info(file_status)
    return vs_path, None, history + [[None, file_status]]


knowledge_base_test_mode_info = ("【注意】\n\n"
                                 "1. 您已进入知识库测试模式，您输入的任何对话内容都将用于进行知识库查询，"
                                 "并仅输出知识库匹配出的内容及相似度分值和及输入的文本源路径，查询的内容并不会进入模型查询。\n\n"
                                 "2. 知识相关度 Score 经测试，建议设置为 500 或更低，具体设置情况请结合实际使用调整。"
                                 """3. 使用"添加单条数据"添加文本至知识库时，内容如未分段，则内容越多越会稀释各查询内容与之关联的score阈值。\n\n"""
                                 "4. 单条内容长度建议设置在100-150左右。")


webui_title = """
# 专业领域大语言模型
"""
######                   #####


###### todo #####
# 1. streamlit运行方式与一般web服务器不同，使用模块是无法实现单例模式的，所以shared和local_doc_qa都需要进行全局化处理。
#   目前已经实现了local_doc_qa和shared.loaderCheckPoint的全局化。
# 2. 当前local_doc_qa是一个全局变量，一方面：任何一个session对其做出修改，都会影响所有session的对话;另一方面，如何处理所有session的请求竞争也是问题。
#   这个暂时无法避免，在配置普通的机器上暂时也无需考虑。
# 3. 目前只包含了get_answer对应的参数，以后可以添加其他参数，如temperature。
######      #####


###### 配置项 #####
class ST_CONFIG:
    default_mode = "知识库问答"
    default_kb = ""
######        #####


class TempFile:
    '''
    为保持与get_vector_store的兼容性，需要将streamlit上传文件转化为其可以接受的方式
    '''

    def __init__(self, path):
        self.name = path


@st.cache_resource(show_spinner=False, max_entries=1)
def load_model(
    llm_model: str = LLM_MODEL,
    embedding_model: str = EMBEDDING_MODEL,
    use_ptuning_v2: bool = USE_PTUNING_V2,
):
    '''
    对应init_model，利用streamlit cache避免模型重复加载
    '''
    local_doc_qa = LocalDocQA()
    # 初始化消息
    args = parser.parse_args()
    args_dict = vars(args)
    args_dict.update(model=llm_model)
    if shared.loaderCheckPoint is None:  # avoid checkpoint reloading when reinit model
        shared.loaderCheckPoint = LoaderCheckPoint(args_dict)
    # shared.loaderCheckPoint.model_name is different by no_remote_model.
    # if it is not set properly error occurs when reinit llm model(issue#473).
    # as no_remote_model is removed from model_config, need workaround to set it automaticlly.
    local_model_path = llm_model_dict.get(llm_model, {}).get('local_model_path') or ''
    no_remote_model = os.path.isdir(local_model_path)
    llm_model_ins = shared.loaderLLM(llm_model, no_remote_model, use_ptuning_v2)
    llm_model_ins.history_len = LLM_HISTORY_LEN

    try:
        local_doc_qa.init_cfg(llm_model=llm_model_ins,
                              embedding_model=embedding_model)
        answer_result_stream_result = local_doc_qa.llm_model_chain(
            {"prompt": "你好", "history": [], "streaming": False})

        for answer_result in answer_result_stream_result['answer_result_stream']:
            print(answer_result.llm_output)
        reply = """模型已成功加载，可以开始对话，或从右侧选择模式后开始对话"""
        logger.info(reply)
    except Exception as e:
        logger.error(e)
        reply = """模型未成功加载，请到页面左上角"模型配置"选项卡中重新选择后点击"加载模型"按钮"""
        if str(e) == "Unknown platform: darwin":
            logger.info("该报错可能因为您使用的是 macOS 操作系统，需先下载模型至本地后执行 Web UI，具体方法请参考项目 README 中本地部署方法及常见问题："
                        " https://github.com/imClumsyPanda/langchain-ChatGLM")
        else:
            logger.info(reply)
    return local_doc_qa


# @st.cache_data
def answer(query, vs_path='', history=[], mode='', score_threshold=0,
           vector_search_top_k=5, chunk_conent=True, chunk_size=100
           ):
    '''
    对应get_answer，--利用streamlit cache缓存相同问题的答案--
    '''
    return get_answer(query, vs_path, history, mode, score_threshold,
                      vector_search_top_k, chunk_conent, chunk_size)


def use_kb_mode(m):
    return m in ["知识库问答", "知识库测试"]


# main ui
st.set_page_config(webui_title, layout='wide')

chat_box = st_chatbox(greetings=["模型已成功加载，可以开始对话，或从左侧选择模式后开始对话。"])
# 使用 help(st_chatbox) 查看自定义参数

# sidebar
modes = ['LLM 对话', '知识库问答', 'Bing搜索问答', '知识库测试']
with st.sidebar:
    def on_mode_change():
        m = st.session_state.mode
        chat_box.robot_say(f'已切换到"{m}"模式')
        if m == '知识库测试':
            chat_box.robot_say(knowledge_base_test_mode_info)

    index = 0
    try:
        index = modes.index(ST_CONFIG.default_mode)
    except:
        pass
    mode = st.selectbox('对话模式', modes, index,
                        on_change=on_mode_change, key='mode')

    with st.expander('模型配置', not use_kb_mode(mode)):
        with st.form('model_config'):
            index = 0
            try:
                index = llm_model_dict_list.index(LLM_MODEL)
            except:
                pass
            llm_model = st.selectbox('LLM模型', llm_model_dict_list, index)

            use_ptuning_v2 = st.checkbox('使用p-tuning-v2微调过的模型', False)
 
            try:
                index = embedding_model_dict_list.index(EMBEDDING_MODEL)
            except:
                pass
            embedding_model = st.selectbox(
                'Embedding模型', embedding_model_dict_list, index)

            btn_load_model = st.form_submit_button('重新加载模型')
            if btn_load_model:
                local_doc_qa = load_model(llm_model, embedding_model, use_ptuning_v2)

    history_len = st.slider(
        "LLM对话轮数", 1, 50, LLM_HISTORY_LEN)

    if use_kb_mode(mode):
        vs_list = get_vs_list()
        vs_list.remove('新建知识库')

        def on_new_kb():
            name = st.session_state.kb_name
            if not name:
                st.sidebar.error(f'新建知识库名称不能为空!')
            elif name in vs_list:
                st.sidebar.error(f'名为“{name}”的知识库已存在。')
            else:
                st.session_state.vs_path = name
                st.session_state.kb_name = ''
                new_kb_dir = os.path.join(KB_ROOT_PATH, name)
                if not os.path.exists(new_kb_dir):
                    os.makedirs(new_kb_dir)
                st.sidebar.success(f'名为“{name}”的知识库创建成功，您可以开始添加文件。')

        def on_vs_change():
            chat_box.robot_say(f'已加载知识库： {st.session_state.vs_path}')
        with st.expander('知识库配置', True):
            cols = st.columns([12, 10])
            kb_name = cols[0].text_input(
                '新知识库名称', placeholder='新知识库名称', label_visibility='collapsed', key='kb_name')
            cols[1].button('新建知识库', on_click=on_new_kb)
            index = 0
            try:
                index = vs_list.index(ST_CONFIG.default_kb)
            except:
                pass
            vs_path = st.selectbox(
                '选择知识库', vs_list, index, on_change=on_vs_change, key='vs_path')

            st.text('')

            score_threshold = st.slider(
                '知识相关度阈值', 0, 1000, VECTOR_SEARCH_SCORE_THRESHOLD)
            top_k = st.slider('向量匹配数量', 1, 20, VECTOR_SEARCH_TOP_K)
            chunk_conent = st.checkbox('启用上下文关联', False)
            chunk_size = st.slider('上下文关联长度', 1, 1000, CHUNK_SIZE)
            st.text('')
            sentence_size = st.slider('文本入库分句长度限制', 1, 1000, SENTENCE_SIZE)
            files = st.file_uploader('上传知识文件',
                                     ['docx', 'txt', 'md', 'csv', 'xlsx', 'pdf'],
                                     accept_multiple_files=True,
                                     )
            if st.button('添加文件到知识库'):
                temp_dir = tempfile.mkdtemp()
                file_list = []
                for f in files:
                    file = os.path.join(temp_dir, f.name)
                    with open(file, 'wb') as fp:
                        fp.write(f.getvalue())
                    file_list.append(TempFile(file))
                _, _, history = get_vector_store(
                vs_path, file_list, sentence_size, [], None, None)
                st.session_state.files = []


# load model after params rendered
with st.spinner(f"正在加载模型({llm_model} + {embedding_model})，请耐心等候..."):
    local_doc_qa = load_model(
        llm_model,
        embedding_model,
        use_ptuning_v2,
    )
    local_doc_qa.llm_model_chain.history_len = history_len
    if use_kb_mode(mode):
        local_doc_qa.chunk_conent = chunk_conent
        local_doc_qa.chunk_size = chunk_size
    # local_doc_qa.llm_model_chain.temperature = temperature # 这样设置temperature似乎不起作用
    st.session_state.local_doc_qa = local_doc_qa

# input form
with st.form("my_form", clear_on_submit=True):
    cols = st.columns([8, 1])
    question = cols[0].text_area(
        'temp', key='input_question', label_visibility='collapsed')

    if cols[1].form_submit_button("发送"):
        chat_box.user_say(question)
        history = []
        if mode == "LLM 对话":
            chat_box.robot_say("正在思考...")
            chat_box.output_messages()
            for history, _ in answer(question,
                                     history=[],
                                     mode=mode):
                chat_box.update_last_box_text(history[-1][-1])
        elif use_kb_mode(mode):
            chat_box.robot_say(f"正在查询 [{vs_path}] ...")
            chat_box.output_messages()
            for history, _ in answer(question,
                                     vs_path=os.path.join(
                                         KB_ROOT_PATH, vs_path, 'vector_store'),
                                     history=[],
                                     mode=mode,
                                     score_threshold=score_threshold,
                                     vector_search_top_k=top_k,
                                     chunk_conent=chunk_conent,
                                     chunk_size=chunk_size):
                chat_box.update_last_box_text(history[-1][-1])
        else:
            chat_box.robot_say(f"正在执行Bing搜索...")
            chat_box.output_messages()
            for history, _ in answer(question,
                                     history=[],
                                     mode=mode):
                chat_box.update_last_box_text(history[-1][-1])

# st.write(chat_box.history)
chat_box.output_messages()
