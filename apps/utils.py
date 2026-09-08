import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st


DATASET_DIR = Path(__file__).resolve().parents[1] / "datasets"


def read_csv_with_fallback(uploaded_file):
    """Read a CSV file with common Chinese encodings."""
    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"):
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc

    raise ValueError(f"无法识别 CSV 文件编码：{last_error}")


def read_dataset_file(uploaded_file):
    if uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        uploaded_file.seek(0)
        return pd.read_excel(uploaded_file, engine="openpyxl")
    return read_csv_with_fallback(uploaded_file)


def dataset_fingerprint(name, df):
    shape = f"{df.shape[0]}x{df.shape[1]}"
    columns = "|".join(map(str, df.columns))
    return hashlib.sha256(f"{name}|{shape}|{columns}".encode("utf-8")).hexdigest()


def clean_dataset(df):
    return df.dropna(how="all").copy()


def set_active_dataset(name, df):
    cleaned_df = clean_dataset(df)
    st.session_state["active_dataset"] = cleaned_df
    st.session_state["dataset_name"] = name
    st.session_state["dataset_hash"] = dataset_fingerprint(name, cleaned_df)


def get_active_dataset():
    df = st.session_state.get("active_dataset")
    if df is None:
        return None, None
    return st.session_state.get("dataset_name", "当前数据集"), df.copy()


def load_sample_dataset(file_name):
    path = DATASET_DIR / file_name
    return pd.read_csv(path)


def render_data_workspace():
    st.subheader("数据工作区")

    sample_files = sorted(path.name for path in DATASET_DIR.glob("*.csv"))
    if sample_files:
        selected_sample = st.selectbox(
            "选择示例数据集",
            ["不使用示例数据"] + sample_files,
            key="workspace_sample_dataset",
        )
        if selected_sample != "不使用示例数据" and st.button("载入示例数据", key="workspace_load_sample"):
            try:
                set_active_dataset(selected_sample, load_sample_dataset(selected_sample))
                st.success(f"已载入：{selected_sample}")
            except Exception as exc:
                st.error(f"载入示例数据失败：{exc}")

    uploaded_file = st.file_uploader(
        "上传全站共享数据",
        type=["csv", "txt", "xlsx"],
        key="workspace_dataset_upload",
    )
    if uploaded_file is not None:
        try:
            set_active_dataset(uploaded_file.name, read_dataset_file(uploaded_file))
            st.success(f"已上传：{uploaded_file.name}")
        except Exception as exc:
            st.error(f"读取数据失败：{exc}")

    dataset_name, active_df = get_active_dataset()
    if active_df is not None:
        st.caption(f"当前数据集：{dataset_name}")
        st.caption(f"{active_df.shape[0]} 行 × {active_df.shape[1]} 列")
        if st.button("清空当前数据集", key="workspace_clear_dataset"):
            for key in ["active_dataset", "dataset_name", "dataset_hash"]:
                st.session_state.pop(key, None)
            st.rerun()


def select_or_upload_dataset(callback_function):
    """使用全站共享数据集；没有共享数据时允许在当前页面临时上传。"""

    dataset_name, active_df = get_active_dataset()
    if active_df is not None:
        st.info(f"正在使用数据工作区的数据集：{dataset_name}")
        callback_function(active_df)
        return

    st.write("请上传你的数据集：")
    data = st.file_uploader("上传数据集", type=["csv", "txt", "xlsx"], key="page_dataset_upload")
    if data is not None:
        try:
            df = read_dataset_file(data)
        except Exception as exc:
            st.error(f"读取数据集失败：{exc}")
            return

        set_active_dataset(data.name, df)
        df = st.session_state["active_dataset"].copy()
        st.success("数据集上传成功")
        callback_function(df)
