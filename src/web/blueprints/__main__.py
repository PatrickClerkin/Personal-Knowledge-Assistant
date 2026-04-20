"""Entry point for running the web server as a module: python -m src.web"""
from .app import app

if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)