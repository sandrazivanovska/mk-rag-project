@echo off
set HF_HOME=D:\hf-cache
set PYTHONIOENCODING=utf-8
cd /d "C:\Users\user\Desktop\nlp\mk-rag-project"
python scripts/mk/17_rescore_ragas.py --target 150 >> results\rescore.log 2>&1
echo RESCORE_FINISHED_EXIT_%ERRORLEVEL% >> results\rescore.log
