import streamlit.web.bootstrap
import sys
import os

# 切换到项目根目录
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置启动参数
port = os.environ.get("PORT", "8501")
args = [
    "streamlit", "run",
    "app.py",
    "--server.port", port,
    "--server.address", "0.0.0.0",
    "--browser.gatherUsageStats", "false",
    "--logger.level", "info"
]

if __name__ == "__main__":
    sys.argv = args
    sys.exit(streamlit.web.bootstrap.run())