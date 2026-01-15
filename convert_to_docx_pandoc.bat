@echo off
REM Convert FINAL_REPORT.md to DOCX using Pandoc
REM This is the recommended method for best formatting

echo ======================================
echo Converting Markdown to DOCX with Pandoc
echo ======================================
echo.

REM Check if Pandoc is installed
where pandoc >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Pandoc is not installed.
    echo.
    echo Install Pandoc from: https://pandoc.org/installing.html
    echo Or use Chocolatey: choco install pandoc
    echo.
    pause
    exit /b 1
)

echo Pandoc found. Converting...
echo.

REM Convert with proper settings
pandoc report\FINAL_REPORT.md ^
    -o report\FINAL_REPORT.docx ^
    --from markdown ^
    --to docx ^
    --number-sections ^
    --toc ^
    --toc-depth=3 ^
    --metadata title="Hur påverkas en omloppsbana av startförhållanden?" ^
    --metadata author="Axel Jönsson" ^
    --metadata date="" ^
    --reference-doc=report_template.docx ^
    --resource-path=report_assets/figures ^
    --standalone

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✓ SUCCESS! Document created at report\FINAL_REPORT.docx
    echo.
    echo NEXT STEPS:
    echo 1. Open report\FINAL_REPORT.docx in Word
    echo 2. Review formatting and figures
    echo 3. Update table of contents ^(right-click → Update Field^)
    echo 4. Add supervisor name on title page
    echo 5. Save and submit!
    echo.
) else (
    echo.
    echo ERROR: Conversion failed.
    echo Check the error message above.
    echo.
)

pause
