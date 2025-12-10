"""
Medical Report Generator using Agentic AI
Run this script to start the Streamlit application.
"""
import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    import streamlit.web.cli as stcli
    
    app_path = os.path.join(os.path.dirname(__file__), 'src', 'app.py')
    sys.argv = ["streamlit", "run", app_path]
    sys.exit(stcli.main())
