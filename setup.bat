@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
set "LOG=%~dp0setup.log"
set "VPY=%~dp0.venv\Scripts\python.exe"
REM Versions the pinned dependencies (PySide6 6.7.3, torch 2.8.0) are tested with.
set "PYCHECK=import struct,sys; ok=(3,10)<=sys.version_info[:2]<=(3,12) and struct.calcsize('P')*8==64; sys.exit(0 if ok else 1)"

> "%LOG%" echo Pisar setup started %date% %time%

echo ================================================
echo   Pisar - ustanovka zavisimostei
echo ================================================
echo   Podrobnyi zhurnal: setup.log
echo.

REM ---------------------------------------------------------------- [1/6] Python
echo [1/6] Poisk Python 3.10-3.12 ^(64-bit^)...
set "PY="
for %%V in (3.12 3.11 3.10) do (
    if not defined PY (
        py -%%V -c "%PYCHECK%" >nul 2>nul
        if not errorlevel 1 set "PY=py -%%V"
    )
)
if not defined PY (
    where python >nul 2>nul
    if not errorlevel 1 (
        python -c "%PYCHECK%" >nul 2>nul
        if not errorlevel 1 set "PY=python"
    )
)
if not defined PY goto no_python

%PY% -c "import sys; print('   Python', sys.version.split()[0], '-', sys.executable)"
%PY% -c "import sys; print(sys.version); print(sys.executable)" >> "%LOG%" 2>&1
%PY% -c "import sys; sys.exit(1 if 'conda' in sys.executable.lower() else 0)" >nul 2>nul
if errorlevel 1 (
    echo.
    echo [VNIMANIE] Eto Python iz Anaconda/Miniconda. S nim chasto voznikayut krashi
    echo ^(DLL load failed, segmentation fault^) iz-za staryh DLL v papke Anaconda.
    echo Luchshe ustanovit' obychnyi Python 3.12 s python.org.
    set "CONT="
    set /p CONT="Prodolzhit' vsyo ravno? [y/n]: "
    if /i not "!CONT!"=="y" exit /b 1
)

REM ---------------------------------------------------------------- [2/6] ffmpeg
echo.
echo [2/6] Proverka ffmpeg...
where ffmpeg >nul 2>nul
if errorlevel 1 (
    echo [VNIMANIE] ffmpeg ne naiden v PATH. Bez nego ne zarabotaet konvertaciya audio.
    echo Ustanovite: winget install Gyan.FFmpeg
    echo ^(ili skachayte s https://www.gyan.dev/ffmpeg/builds/ i dobav'te papku bin v PATH^)
    echo Posle ustanovki perezapustite terminal.
    echo.
    pause
)

REM ---------------------------------------------------------------- [3/6] venv
echo.
echo [3/6] Virtual'noe okruzhenie .venv...
if exist "%VPY%" (
    "%VPY%" -c "%PYCHECK%" >nul 2>nul
    if errorlevel 1 (
        echo    Sushchestvuyushchii .venv ne podkhodit ^(drugaya versiya Python ili povrezhden^) - sozdayu zanovo.
        rmdir /s /q .venv
    ) else (
        echo    .venv uzhe est i podkhodit.
    )
) else (
    if exist .venv rmdir /s /q .venv
)
if not exist "%VPY%" (
    %PY% -m venv .venv >> "%LOG%" 2>&1
    if errorlevel 1 (
        set "FAILMSG=Ne udalos' sozdat' .venv"
        goto fail
    )
)

REM ---------------------------------------------------------------- [4/6] pip
echo.
echo [4/6] Obnovlenie pip...
"%VPY%" -m pip install --upgrade pip --disable-pip-version-check >> "%LOG%" 2>&1
if errorlevel 1 (
    set "FAILMSG=Ne udalos' obnovit' pip"
    goto fail
)

REM ---------------------------------------------------------------- [5/6] requirements
echo.
echo [5/6] Ustanovka bibliotek iz requirements.txt...
echo    Eto mozhet zanyat' 5-15 minut, okno ne zavislo - idet zagruzka.
"%VPY%" -m pip install -r requirements.txt --disable-pip-version-check >> "%LOG%" 2>&1
if errorlevel 1 (
    set "FAILMSG=Ne udalos' ustanovit' biblioteki iz requirements.txt"
    goto fail
)

REM ---------------------------------------------------------------- [6/6] PyTorch
REM requirements.txt already installed torch 2.8.0 from PyPI, which is the CPU
REM build. For an NVIDIA GPU swap in the CUDA build of the SAME version;
REM --no-deps so pip does not re-resolve (and touch) anything else.
echo.
echo [6/6] PyTorch...
set "DEFAULT_GPU=n"
where nvidia-smi >nul 2>nul && set "DEFAULT_GPU=y"
set "HASGPU="
set /p HASGPU="Videokarta NVIDIA - ustanovit' PyTorch s CUDA? [y/n, Enter = %DEFAULT_GPU%]: "
if not defined HASGPU set "HASGPU=%DEFAULT_GPU%"
if /i "!HASGPU!"=="y" (
    echo    PyTorch s CUDA - eto samyi dolgii shag ^(okolo 2.5 GB^)...
    "%VPY%" -m pip install --force-reinstall --no-deps torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126 --disable-pip-version-check >> "%LOG%" 2>&1
    if errorlevel 1 (
        set "FAILMSG=Ne udalos' ustanovit' PyTorch s CUDA"
        goto fail
    )
) else (
    echo    Ostavlyayu PyTorch dlya CPU - on uzhe ustanovlen.
)

REM ---------------------------------------------------------------- check
echo.
echo Proverka ustanovki...
"%VPY%" -c "import torch; print('   PyTorch', torch.__version__, '- CUDA dostupna:', torch.cuda.is_available())" 2>> "%LOG%"
"%VPY%" -c "import faster_whisper, pyannote.audio, PySide6, sympy; print('   faster-whisper, pyannote.audio, PySide6 - OK; sympy', sympy.__version__)" 2>> "%LOG%"
if errorlevel 1 (
    set "FAILMSG=Odna iz osnovnyh bibliotek ne importiruetsya"
    goto fail
)

echo.
echo ================================================
echo   Gotovo. Zapuskajte prilozhenie: run.bat
echo ================================================
pause
exit /b 0

:no_python
echo.
echo [OSHIBKA] Ne nayden podkhodyashchii Python.
echo Nuzhen 64-bitnyi Python 3.10, 3.11 ili 3.12 - imenno na nih proverena rabota.
echo Ustanovite Python 3.12 s https://www.python.org/downloads/windows/ i otmet'te
echo galochki "Add python.exe to PATH" i "py launcher". Zatem zapustite setup.bat snova.
echo.
echo Naideno na etom komp'yutere:
where py >nul 2>nul && py -0p
python --version 2>nul
echo.
pause
exit /b 1

:fail
echo.
echo [OSHIBKA] %FAILMSG%
echo Poslednie stroki zhurnala ^(polnyi zhurnal - v fayle setup.log^):
echo ------------------------------------------------
powershell -NoProfile -Command "Get-Content -LiteralPath '%LOG%' -Tail 25"
echo ------------------------------------------------
echo Esli ne poluchaetsya razobrat'sya - otpravte fail setup.log razrabotchiku.
pause
exit /b 1
