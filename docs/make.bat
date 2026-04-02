@ECHO OFF
set SOURCEDIR=source
set BUILDDIR=build

if "%1"=="" goto help

python -m sphinx -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%
goto end

:help
echo.
echo Usage: make.bat html
echo.

:end
