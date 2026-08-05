<#
.SYNOPSIS
    Build script for LaTeX research paper.
.DESCRIPTION
    Compiles main.tex using latexmk or pdflatex into the build/ directory,
    and copies the final PDF to main.pdf.
.PARAMETER Clean
    Removes auxiliary build files and build/ directory.
.PARAMETER Watch
    Continuously monitors TeX files and recompiles on change (latexmk -pvc).
.EXAMPLE
    .\build.ps1
    .\build.ps1 -Clean
    .\build.ps1 -Watch
#>

param(
    [switch]$Clean,
    [switch]$Watch
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$BuildDir = Join-Path $ProjectRoot "build"
$MainTex = Join-Path $ProjectRoot "main.tex"
$OutputPdf = Join-Path $BuildDir "main.pdf"
$RootPdf = Join-Path $ProjectRoot "main.pdf"

if ($Clean) {
    Write-Host "Cleaning build directory..." -ForegroundColor Yellow
    if (Test-Path $BuildDir) {
        Remove-Item -Recurse -Force $BuildDir
    }
    if (Test-Path $RootPdf) {
        Remove-Item -Force $RootPdf
    }
    Write-Host "Clean complete." -ForegroundColor Green
    exit 0
}

# Ensure build directory exists
if (-not (Test-Path $BuildDir)) {
    New-Item -ItemType Directory -Path $BuildDir | Out-Null
}

# Check for latexmk
$hasLatexmk = Get-Command "latexmk" -ErrorAction SilentlyContinue

if ($Watch) {
    if ($hasLatexmk) {
        Write-Host "Starting continuous watch mode with latexmk..." -ForegroundColor Cyan
        Set-Location $ProjectRoot
        latexmk -pdf -pvc -outdir=build main.tex
    } else {
        Write-Error "latexmk is required for watch mode (-Watch)."
    }
    exit 0
}

Write-Host "Compiling main.tex..." -ForegroundColor Cyan

if ($hasLatexmk) {
    Set-Location $ProjectRoot
    latexmk -pdf -outdir=build main.tex
} else {
    Write-Host "latexmk not found; falling back to direct pdflatex/bibtex pipeline..." -ForegroundColor Yellow
    Set-Location $ProjectRoot
    pdflatex -interaction=nonstopmode -output-directory=build main.tex
    bibtex build/main
    pdflatex -interaction=nonstopmode -output-directory=build main.tex
    pdflatex -interaction=nonstopmode -output-directory=build main.tex
}

if (Test-Path $OutputPdf) {
    Copy-Item -Path $OutputPdf -Destination $RootPdf -Force
    Write-Host "Compilation successful!" -ForegroundColor Green
    Write-Host "Output PDF: $RootPdf" -ForegroundColor Green
} else {
    Write-Error "Compilation failed; output PDF not found."
}
