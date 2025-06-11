@echo off
echo Retraining with 4-Class Dataset
echo ================================
cd /d e:\5g_miniproject
call venv\Scripts\activate.bat
echo.
echo Dataset Configuration:
echo - Classes: male, female, injury, no injury
echo - Training with corrected data.yaml
echo.
python train_injury_model.py --epochs 100 --batch-size 16 --data-yaml accident-image.v1i.yolov8/data.yaml --device 0
pause
