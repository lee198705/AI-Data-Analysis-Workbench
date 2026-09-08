import streamlit as st
from multiapp import MultiApp
from apps.utils import render_data_workspace

st.set_page_config(page_title="小鱼实验室", layout="wide")

if __name__ == "__main__":
    from apps import analy, chatting, eda, models

    app = MultiApp()

    # 将各个 app 添加到多应用管理器中
    app.add_app("数据探索", eda.app)
    app.add_app("机器学习模型", models.app)
    app.add_app("智能问答系统", chatting.app)
    app.add_app("数据可视化", analy.app)

    # 在页面顶部添加标题、描述和图标
    st.title("实验室")
    st.image("picture/OIP1.jpg", width=100)  # 替换为实际的图像路径
    st.markdown("欢迎来到实验室，这是一个集数据探索、机器学习、智能问答和数据可视化于一体的平台。")

    # 添加一个侧边栏用于导航
    with st.sidebar:
        st.header("导航")
        st.image("picture/OIP.jpg", width=150)  # 替换为实际的图像路径
        st.write("请从以下选项中选择一个应用程序：")
        render_data_workspace()

    app.run()
