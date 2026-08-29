@echo off
set HF_HOME=D:\hf-cache
set PYTHONIOENCODING=utf-8
cd /d "C:\Users\user\Desktop\nlp\mk-rag-project"
python scripts/mk/18_second_generator.py --model gemini-2.5-flash-lite >> results\gen2.log 2>&1
echo GEN2_FINISHED_EXIT_%ERRORLEVEL% >> results\gen2.log
