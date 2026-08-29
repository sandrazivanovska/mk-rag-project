@echo off
set HF_HOME=D:\hf-cache
set PYTHONIOENCODING=utf-8
cd /d "C:\Users\user\Desktop\nlp\mk-rag-project"
python main.py run-all --gold-path data/evaluation/gold_dataset_crosslingual.jsonl --generators gemini_flash --output-dir results/full >> results\full_run.log 2>&1
echo RUN_FINISHED_EXIT_%ERRORLEVEL% >> results\full_run.log
