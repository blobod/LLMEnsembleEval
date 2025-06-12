@echo off
setlocal EnableDelayedExpansion

:: Create log file with timestamp
set "timestamp=%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
set "timestamp=%timestamp: =0%"
set "logfile=gac_evaluation_%timestamp%.log"

echo Starting GAC Evaluation at %date% %time%
echo Starting GAC Evaluation at %date% %time% > %logfile%
echo ================================================
echo ================================================ >> %logfile%

:: Check if required files exist
echo Checking for required files...
echo Checking for required files... >> %logfile%
if not exist "ensemble.py" (
    echo ERROR: ensemble.py not found in current directory
    echo ERROR: ensemble.py not found in current directory >> %logfile%
    goto :error
)
echo Found: ensemble.py
echo Found: ensemble.py >> %logfile%

if not exist "wrapper.py" (
    echo ERROR: wrapper.py not found in current directory
    echo ERROR: wrapper.py not found in current directory >> %logfile%
    goto :error
)
echo Found: wrapper.py
echo Found: wrapper.py >> %logfile%

if not exist "config.json" (
    echo ERROR: config.json not found in current directory
    echo ERROR: config.json not found in current directory >> %logfile%
    goto :error
)
echo Found: config.json
echo Found: config.json >> %logfile%

::Clone repository (only if it doesn't exist)
echo.
echo. >> %logfile%
if exist "lm-evaluation-harness" (
    echo lm-evaluation-harness directory already exists, skipping clone...
    echo lm-evaluation-harness directory already exists, skipping clone... >> %logfile%
) else (
    echo Cloning lm-evaluation-harness...
    echo Cloning lm-evaluation-harness... >> %logfile%
    git clone https://github.com/EleutherAI/lm-evaluation-harness.git >> %logfile% 2>&1
    if errorlevel 1 (
        echo ERROR: Failed to clone repository
        echo ERROR: Failed to clone repository >> %logfile%
        goto :error
    )
    echo Successfully cloned repository
    echo Successfully cloned repository >> %logfile%
)

:: Install lm-evaluation-harness
echo.
echo. >> %logfile%
echo Installing lm-evaluation-harness...
echo Installing lm-evaluation-harness... >> %logfile%
pip install -e lm-evaluation-harness >> %logfile% 2>&1
if errorlevel 1 (
    echo ERROR: Failed to install lm-evaluation-harness
    echo ERROR: Failed to install lm-evaluation-harness >> %logfile%
    goto :error
)
echo Successfully installed lm-evaluation-harness
echo Successfully installed lm-evaluation-harness >> %logfile%

:: Test imports and register the model
echo.
echo. >> %logfile%
echo Testing imports and registering model...
echo Testing imports and registering model... >> %logfile%
python -c "from wrapper import EnsembleHarnessWrapper; from lm_eval.api.registry import MODEL_REGISTRY; print('Wrapper imported and registered successfully'); print('gac_ensemble_wrapper registered:', 'gac_ensemble_wrapper' in MODEL_REGISTRY)" >> %logfile% 2>&1
if errorlevel 1 (
    echo ERROR: Failed to import and register wrapper
    echo ERROR: Failed to import and register wrapper >> %logfile%
    goto :error
)
echo Import and registration test successful
echo Import and registration test successful >> %logfile%

:: Run evaluation from current directory
echo.
echo. >> %logfile%
echo Running evaluation...
echo Running evaluation... >> %logfile%
echo Creating wrapper script to ensure registration...
echo Creating wrapper script to ensure registration... >> %logfile%

:: Create a temporary Python script that imports the wrapper and runs lm_eval
echo import sys > run_eval_temp.py
echo from wrapper import EnsembleHarnessWrapper >> run_eval_temp.py
echo from lm_eval.__main__ import cli_evaluate >> run_eval_temp.py
echo import json >> run_eval_temp.py
echo try: >> run_eval_temp.py
echo     with open('config.json', 'r') as f: config = json.load(f) >> run_eval_temp.py
echo     benchmark = config['benchmark'] >> run_eval_temp.py
echo except KeyError: >> run_eval_temp.py
echo     print('ERROR: benchmark field is required in config.json') >> run_eval_temp.py
echo     print('Supported benchmarks: piqa, mmlu, arc_challenge, winogrande') >> run_eval_temp.py
echo     exit(1) >> run_eval_temp.py
echo sys.argv = ['lm_eval', '--model', 'gac_ensemble_wrapper', '--model_args', 'config_path=config.json', '--tasks', benchmark, '--batch_size', '1', '--num_fewshot', '0', '--output_path', 'results'] >> run_eval_temp.py
echo cli_evaluate() >> run_eval_temp.py

echo Command: python run_eval_temp.py
echo Command: python run_eval_temp.py >> %logfile%

python run_eval_temp.py >> %logfile% 2>&1

:: Clean up temporary file
del run_eval_temp.py 2>nul

if errorlevel 1 (
    echo ERROR: Evaluation failed
    echo ERROR: Evaluation failed >> %logfile%
    goto :error
) else (
    echo.
    echo. >> %logfile%
    echo SUCCESS: Evaluation completed!
    echo SUCCESS: Evaluation completed! >> %logfile%
    echo Results saved to results/
    echo Results saved to results/ >> %logfile%
    echo Log saved to %logfile%
    echo Log saved to %logfile% >> %logfile%
)

goto :end

:error
echo.
echo. >> %logfile%
echo ================================================
echo ================================================ >> %logfile%
echo SCRIPT FAILED - Check the log above for details
echo SCRIPT FAILED - Check the log above for details >> %logfile%
echo Log saved to: %logfile%
echo Log saved to: %logfile% >> %logfile%
echo ================================================
echo ================================================ >> %logfile%
echo.
echo Press any key to exit...
pause >nul
exit /b 1

:end
echo.
echo Script completed successfully!
echo Press any key to exit...
pause >nul