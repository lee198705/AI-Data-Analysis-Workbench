"""Frameworks for running multiple Streamlit applications as a single app."""
import streamlit as st

class MultiApp:
    """Framework for combining multiple streamlit applications."""

    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        """
        Adds a new application.

        Parameters:
        -----------
        title (str): Title of the app. Appears in the sidebar menu.
        func (python function): The function to render this app.
        """
        self.apps.append({
            "title": title,
            "function": func
        })

    def run(self):
        # 在侧边栏添加应用程序导航菜单

        selection = st.sidebar.radio("转到", [app["title"] for app in self.apps])
        # 根据用户选择运行相应的应用程序
        for app in self.apps:
            if app["title"] == selection:
                app["function"]()
