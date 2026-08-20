@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ================================================
echo   Whisper Diarizer - установка зависимостей
echo ================================================
echo.

REM --- Python check ---
where python >nul 2>nul
if errorlevel 1 (
    echo [OSHIBKA] Python ne naiden v PATH.
    echo Ustanovite Python 3.10-3.12 s https://www.python.org/downloads/
    echo VAZHNO: pri ustanovke otmet'te "Add python.exe to PATH".
    echo NE ispol'zuite Anaconda/Miniconda - sm. README.md, razdel pro DLL-konflikty.
    pause
    exit /b 1
)

python -c "import sys; sys.exit(1 if ('anaconda' in sys.executable.lower() or 'miniconda' in sys.executable.lower() or 'conda' in sys.executable.lower()) else 0)"
if errorlevel 1 (
    echo [VNIMANIE] Pokhozhe, eto Python iz Anaconda/Miniconda: %errorlevel%
    echo Eto chasto vyzyvaet krashi (DLL load failed / segmentation fault^)
    echo iz-za starykh biblioteknyh faylov v papke Anaconda.
    echo Rekomenduetsya ustanovit' "obychnyi" Python s python.org i zapustit'
    echo etot skript im (ukazhite polnyi put' k python.exe pri zapuske,
    echo ili sdelaite ego pervym v PATH^).
    echo.
    set /p CONTINUE="Prodolzhit' vsyo ravno? (y/n): "
    if /i not "!CONTINUE!"=="y" (
        exit /b 1
    )
)

echo [1/5] Proverka ffmpeg...
where ffmpeg >nul 2>nul
if errorlevel 1 (
    echo [VNIMANIE] ffmpeg ne naiden v PATH.
    echo Skachaite s https://www.gyan.dev/ffmpeg/builds/ ^(release full^)
    echo i dobav'te papku bin v PATH. Bez nego ne zarabotaet konvertaciya audio.
    echo.
    pause
)

echo [2/5] Sozdanie virtual'nogo okruzheniya .venv...
if exist .venv (
    echo .venv uzhe sushchestvuet, propuskaem sozdanie.
) else (
    python -m venv .venv
    if errorlevel 1 (
        echo [OSHIBKA] Ne udalos' sozdat' .venv
        pause
        exit /b 1
    )
)

echo [3/5] Obnovlenie pip...
.venv\Scripts\python.exe -m pip install --upgrade pip -q

echo [4/5] Ustanovka zavisimostei iz requirements.txt...
.venv\Scripts\python.exe -m pip install -r requirements.txt -q
if errorlevel 1 (
    echo [OSHIBKA] Ne udalos' ustanovit' zavisimosti iz requirements.txt
    pause
    exit /b 1
)

echo [5/5] Ustanovka PyTorch...
echo.
echo Est' li na etom komp'yutere videokarta NVIDIA ^(dlya uskoreniya na GPU^)?
set /p HASGPU="Vvedite y (est' NVIDIA GPU) ili n (net / ispol'zovat' CPU): "
if /i "!HASGPU!"=="y" (
    echo Ustanavlivayu PyTorch s podderzhkoi CUDA...
    .venv\Scripts\python.exe -m pip install --force-reinstall --no-deps torch torchaudio --index-url https://download.pytorch.org/whl/cu124 -q
    if errorlevel 1 (
        echo [OSHIBKA] Ne udalos' ustanovit' PyTorch s CUDA.
        pause
        exit /b 1
    )
) else (
    echo Ustanavlivayu PyTorch ^(CPU^)...
    .venv\Scripts\python.exe -m pip install --force-reinstall --no-deps torch torchaudio -q
    if errorlevel 1 (
        echo [OSHIBKA] Ne udalos' ustanovit' PyTorch.
        pause
        exit /b 1
    )
)

echo.
echo Proverka ustanovki...
.venv\Scripts\python.exe -c "import torch; print('PyTorch', torch.__version__, '| CUDA dostupna:', torch.cuda.is_available())"
.venv\Scripts\python.exe -c "import faster_whisper, pyannote.audio, PySide6; print('faster-whisper, pyannote.audio, PySide6 - importiruyutsya uspeshno')"
if errorlevel 1 (
    echo.
    echo [OSHIBKA] Odna iz osnovnykh bibliotek ne importiruetsya. Sm. tekst oshibki vyshe.
    pause
    exit /b 1
)

echo.
echo ================================================
echo   Gotovo! Zapuskaite prilozhenie: run.bat
echo ================================================
pause
