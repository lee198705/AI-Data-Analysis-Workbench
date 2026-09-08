import hashlib
import io

import pandas as pd
import streamlit as st
from docx import Document
from PyPDF2 import PdfReader

from apps.utils import get_active_dataset, read_csv_with_fallback
from chain import generate_response
from rag import build_rag_prompt, chunks_from_dataframe, chunks_from_text, retrieve_context


SUPPORTED_FILE_TYPES = ["csv", "xlsx", "pdf", "txt", "docx"]


def init_chat_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "document_chunks" not in st.session_state:
        st.session_state.document_chunks = []
    if "uploaded_file_hash" not in st.session_state:
        st.session_state.uploaded_file_hash = None
    if "uploaded_file_name" not in st.session_state:
        st.session_state.uploaded_file_name = None
    if "active_dataset_chunks" not in st.session_state:
        st.session_state.active_dataset_chunks = []
    if "active_dataset_hash_for_chat" not in st.session_state:
        st.session_state.active_dataset_hash_for_chat = None


def read_uploaded_file(uploaded_file):
    if uploaded_file.type == "text/csv":
        return read_csv_with_fallback(uploaded_file)

    if uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        return pd.read_excel(uploaded_file, engine="openpyxl")

    if uploaded_file.type == "application/pdf":
        pdf = PdfReader(uploaded_file)
        pages = []
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(f"第 {page_number} 页\n{text}")
        return "\n\n".join(pages)

    if uploaded_file.type == "text/plain":
        stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
        return stringio.read()

    if uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(uploaded_file)
        return "\n".join(paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip())

    raise ValueError("暂不支持该文件类型。")


def hash_uploaded_file(uploaded_file):
    uploaded_file.seek(0)
    digest = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
    uploaded_file.seek(0)
    return digest


def update_uploaded_context(uploaded_file, file_contents):
    if isinstance(file_contents, pd.DataFrame):
        st.session_state.document_chunks = chunks_from_dataframe(
            file_contents,
            source=f"上传文件：{uploaded_file.name}",
        )
    else:
        st.session_state.document_chunks = chunks_from_text(
            file_contents,
            source=f"上传文件：{uploaded_file.name}",
        )
    st.session_state.uploaded_file_name = uploaded_file.name
    st.session_state.uploaded_file_hash = hash_uploaded_file(uploaded_file)


def render_uploaded_file(uploaded_file):
    if uploaded_file is None:
        st.session_state.document_chunks = []
        st.session_state.uploaded_file_hash = None
        st.session_state.uploaded_file_name = None
        return

    current_file_hash = hash_uploaded_file(uploaded_file)
    if st.session_state.uploaded_file_hash == current_file_hash:
        return

    try:
        file_contents = read_uploaded_file(uploaded_file)
    except Exception as exc:
        st.error(f"读取文件失败：{exc}")
        return

    update_uploaded_context(uploaded_file, file_contents)

    st.success(f"已加入检索上下文：{uploaded_file.name}")
    if isinstance(file_contents, pd.DataFrame):
        st.dataframe(file_contents.head(200), width="stretch")
    else:
        st.text_area("文件预览", file_contents[:8000], height=260)


def sync_active_dataset_context():
    dataset_name, active_df = get_active_dataset()
    dataset_hash = st.session_state.get("dataset_hash")
    if active_df is None or dataset_hash is None:
        st.session_state.active_dataset_chunks = []
        st.session_state.active_dataset_hash_for_chat = None
        return

    if st.session_state.active_dataset_hash_for_chat == dataset_hash:
        return

    st.session_state.active_dataset_chunks = chunks_from_dataframe(
        active_df,
        source=f"数据工作区：{dataset_name}",
    )
    st.session_state.active_dataset_hash_for_chat = dataset_hash


def get_chat_chunks():
    sync_active_dataset_context()
    return st.session_state.document_chunks + st.session_state.active_dataset_chunks


def render_sources(results):
    if not results:
        st.caption("未检索到可引用来源。")
        return

    with st.expander("查看本次回答引用的资料来源"):
        for index, result in enumerate(results, start=1):
            st.markdown(f"**来源 {index}** · {result.chunk.source} · score={result.score:.3f}")
            st.caption(result.chunk.content[:260])


def app():
    st.title("统计聊天机器人")
    st.caption("支持本地统计知识库和上传文档检索。")
    init_chat_state()

    uploaded_file = st.file_uploader("上传文档", type=SUPPORTED_FILE_TYPES)
    render_uploaded_file(uploaded_file)
    sync_active_dataset_context()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("请输入统计学问题或针对上传文档提问", submit_mode="disable")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("正在检索资料并生成回答..."):
            results = retrieve_context(prompt, get_chat_chunks(), top_k=5)
            rag_prompt = build_rag_prompt(prompt, results)
            try:
                answer = generate_response(rag_prompt, st.session_state.messages[:-1])
            except RuntimeError as exc:
                answer = f"配置错误：{exc}"
            except Exception as exc:
                answer = f"生成回答失败：{exc}"
        st.markdown(answer)
        render_sources(results)

    st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    app()
