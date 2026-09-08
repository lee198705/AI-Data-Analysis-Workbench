import matplotlib
import pandas as pd
import streamlit as st

from .plotting import configure_chinese_font
from .utils import select_or_upload_dataset

matplotlib.use("Agg")
configure_chinese_font()
matplotlib.rcParams.update({"font.size": 9})
import matplotlib.pyplot as plt
import seaborn as sns


def categorical_column(df, max_unique_values=15):
    categorical_column_list = []
    for column in df.columns:
        if df[column].nunique() < max_unique_values:
            categorical_column_list.append(column)
    return categorical_column_list


def eda(df):

    if st.checkbox("查看数据集"):
        number = st.number_input("查看行数", min_value=1, value=5)
        st.dataframe(df.head(number))

    if st.checkbox("查看字段名称"):
        st.write(df.columns)

    if st.checkbox("查看数据集形状"):
        st.write(df.shape)
        data_dim = st.radio("选择维度", ("行数", "列数"), key="eda_dimension")
        if data_dim == "列数":
            st.text("列数")
            st.write(df.shape[1])
        elif data_dim == "行数":
            st.text("行数")
            st.write(df.shape[0])
        else:
            st.write(df.shape)

    if st.checkbox("选择字段查看"):
        all_columns = df.columns.tolist()
        selected_columns = st.multiselect("选择字段", all_columns, key="eda_preview_columns")
        if selected_columns:
            st.dataframe(df[selected_columns])

    if st.checkbox("查看字段取值计数"):
        all_columns = df.columns.tolist()
        selected_columns = st.selectbox("选择字段", all_columns, key="eda_value_count_column")
        st.write(df[selected_columns].value_counts())

    if st.checkbox("查看字段类型"):
        st.text("字段类型")
        dtype_df = pd.DataFrame(
            {"字段": df.columns, "类型": [str(dtype) for dtype in df.dtypes]}
        )
        st.dataframe(dtype_df, hide_index=True)

    if st.checkbox("查看统计摘要"):
        st.text("统计摘要")
        st.write(df.describe().T)

    st.subheader("数据可视化")
    all_columns_names = df.columns.tolist()

    if st.checkbox("显示相关性热力图"):
        numeric_df = df.select_dtypes(include="number")
        if numeric_df.shape[1] < 2:
            st.warning("相关性热力图至少需要两个数值型字段。")
        else:
            st.success("正在生成相关性热力图...")
            fig_width = max(10, min(18, numeric_df.shape[1] * 0.7))
            fig_height = max(7, min(18, numeric_df.shape[1] * 0.55))
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            if st.checkbox("在图中显示数值", key="eda_corr_annot"):
                sns.heatmap(numeric_df.corr(), annot=True, ax=ax)
            else:
                sns.heatmap(numeric_df.corr(), ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    if st.checkbox("显示取值计数图"):
        x = st.selectbox("选择分类字段", all_columns_names, key="eda_count_x")
        st.success("正在生成图表...")
        if x:
            fig, ax = plt.subplots()
            if st.checkbox("选择第二个分类字段", key="eda_count_hue_enabled"):
                hue_all_column_name = df[df.columns.difference([x])].columns
                hue = st.selectbox("选择分组字段", hue_all_column_name, key="eda_count_hue")
                sns.countplot(x=x, hue=hue, data=df, palette="Set2", ax=ax)
            else:
                sns.countplot(x=x, data=df, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    if st.checkbox("显示饼图"):
        all_columns = categorical_column(df)
        selected_columns = st.selectbox("选择字段", all_columns, key="eda_pie_column")
        if selected_columns:
            st.success("正在生成饼图...")
            fig, ax = plt.subplots()
            df[selected_columns].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    st.subheader("自定义图表")

    plot_type_labels = {
        "面积图": "area",
        "柱状图": "bar",
        "折线图": "line",
        "直方图": "hist",
        "箱线图": "box",
        "核密度图": "kde",
    }
    type_of_plot = st.selectbox(
        "选择图表类型", list(plot_type_labels.keys()), key="eda_custom_plot_type"
    )
    selected_columns_names = st.multiselect("选择绘图字段", all_columns_names, key="eda_custom_columns")

    if st.button("生成图表"):
        if not selected_columns_names:
            st.warning("请至少选择一个字段。")
            return

        plot_kind = plot_type_labels[type_of_plot]
        st.success(f"正在生成 {type_of_plot}：{selected_columns_names}")
        custom_data = df[selected_columns_names]
        if plot_kind == "area":
            st.area_chart(custom_data)

        elif plot_kind == "bar":
            st.bar_chart(custom_data)

        elif plot_kind == "line":
            st.line_chart(custom_data)

        elif plot_kind:
            fig, ax = plt.subplots()
            df[selected_columns_names].plot(kind=plot_kind, ax=ax)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # st.balloons()


def app():

    st.title("数据探索")

    select_or_upload_dataset(eda)


if __name__ == "__main__":
    app()
