@echo off
if "%~1"=="" (
    echo Please provide a commit message.
    exit /b
)

git add .
git commit -m "%~1"
git push myfork main
git push hpcc-fork main
