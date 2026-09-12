[CmdletBinding()]
param(
    [string[]]$Languages = @("jpn", "deu", "eng")
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$tessdataDir = Join-Path $repoRoot ".venv\tools\tessdata"

if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    throw "Project Python not found: $venvPython"
}

$tesseractCandidates = @(
    (Get-Command tesseract -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -First 1),
    (Join-Path $env:ProgramFiles "Tesseract-OCR\tesseract.exe"),
    (Join-Path $env:LOCALAPPDATA "Programs\Tesseract-OCR\tesseract.exe")
) | Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Leaf) }

$tesseract = $tesseractCandidates | Select-Object -First 1
if (-not $tesseract) {
    throw "Tesseract is missing. Install it first with: winget install --exact --id tesseract-ocr.tesseract"
}

& $venvPython -m pip install pypdf pymupdf
if ($LASTEXITCODE -ne 0) {
    throw "Python dependency installation failed with exit code $LASTEXITCODE"
}

New-Item -ItemType Directory -Force -Path $tessdataDir | Out-Null
foreach ($language in $Languages) {
    if ($language -notmatch "^[a-z0-9_]+$") {
        throw "Invalid Tesseract language identifier: $language"
    }
    $url = "https://github.com/tesseract-ocr/tessdata_fast/raw/main/$language.traineddata"
    $target = Join-Path $tessdataDir "$language.traineddata"
    Write-Host "Downloading $language to $target"
    Invoke-WebRequest -Uri $url -OutFile $target
}

& $tesseract --tessdata-dir $tessdataDir --list-langs
if ($LASTEXITCODE -ne 0) {
    throw "Tesseract language validation failed with exit code $LASTEXITCODE"
}

Write-Host "OCR setup complete. Tessdata: $tessdataDir"
