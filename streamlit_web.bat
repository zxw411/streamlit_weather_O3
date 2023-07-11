chcp 65001
@echo off
call activate 
call conda activate web
streamlit run my_streamlit.py
Pause
