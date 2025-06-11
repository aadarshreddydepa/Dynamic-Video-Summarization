@echo off
set PYTHONPATH=%PYTHONPATH%;%CD%
python src/main.py infer --checkpoint_path models/best_model.pt