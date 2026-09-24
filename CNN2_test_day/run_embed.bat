@echo off
REM Test day: run from THIS folder (CNN2\test_day). Put the given test_dataset folder here first.
REM Uses model.pt in this folder if present, otherwise ..\model.pt (the CNN2 repo copy).
if exist model.pt (set MODEL=model.pt) else (set MODEL=..\model.pt)
python predict_test_embed.py --model %MODEL% --test-dir test_dataset --group NPC
echo.
echo Next: open predictions_NPC.csv.values.txt, copy ALL lines into the NPC column of the sheet.
pause
