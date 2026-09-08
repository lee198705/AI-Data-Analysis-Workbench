import pandas as pd
import streamlit as st

from apps.utils import get_active_dataset, read_csv_with_fallback


def read_uploaded_dataset(uploaded_file):
    if uploaded_file.type == "text/csv":
        return read_csv_with_fallback(uploaded_file)
    return pd.read_excel(uploaded_file, engine="openpyxl")


def render_native_charts(df):
    st.subheader("基础可视化")
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    all_columns = df.columns.tolist()

    if not all_columns:
        st.warning("数据集中没有可用字段。")
        return

    chart_type = st.selectbox("选择图表类型", ["折线图", "柱状图", "面积图", "散点图"])

    if chart_type in ["折线图", "柱状图", "面积图"]:
        selected_columns = st.multiselect("选择数值字段", numeric_columns, default=numeric_columns[:2])
        if not selected_columns:
            st.warning("请至少选择一个数值字段。")
            return

        if chart_type == "折线图":
            st.line_chart(df[selected_columns])
        elif chart_type == "柱状图":
            st.bar_chart(df[selected_columns])
        else:
            st.area_chart(df[selected_columns])
        return

    if len(numeric_columns) < 2:
        st.warning("散点图至少需要两个数值字段。")
        return

    x_column = st.selectbox("选择 X 轴字段", numeric_columns)
    y_options = [column for column in numeric_columns if column != x_column]
    y_column = st.selectbox("选择 Y 轴字段", y_options)
    st.scatter_chart(df, x=x_column, y=y_column)


def app():
    st.title("数据可视化")
    dataset_name, active_df = get_active_dataset()
    if active_df is not None:
        st.info(f"正在使用数据工作区的数据集：{dataset_name}")
        df = active_df
    else:
        uploaded_file = st.file_uploader("请上传数据", type=["csv", "xlsx"])
        if uploaded_file is None:
            return

        try:
            df = read_uploaded_dataset(uploaded_file)
        except Exception as exc:
            st.error(f"读取数据失败：{exc}")
            return

    df.dropna(how="all", inplace=True)
    st.success("数据已准备就绪")
    st.dataframe(df.head(200), width="stretch")

    try:
        from pygwalker.api.streamlit import StreamlitRenderer
    except ImportError as exc:
        st.warning("Pygwalker 与当前 Streamlit 版本不兼容，已切换为内置可视化。")
        st.caption(f"错误信息：{exc}")
        render_native_charts(df)
        return

    try:
        pyg_add = StreamlitRenderer(df)
        pyg_add.explorer()
    except Exception as exc:
        st.warning("Pygwalker 加载失败，已切换为内置可视化。")
        st.caption(f"错误信息：{exc}")
        render_native_charts(df)


if __name__ == "__main__":
    app()

