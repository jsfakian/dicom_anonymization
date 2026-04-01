[CmdletBinding()]
param(
    [switch]$Clean,
    [switch]$SkipRequirements,
    [switch]$UseSpec = $true,
    [string]$VenvDir = ".venv_build",
    [string]$RequirementsFile = "requirements-build.txt",
    [string]$SpecFile = "DICOM-DeID.spec",
    [string]$EntryScript = "anonymizer_pro.py",
    [string]$ExeName = "DICOM-DeID"
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "=== $Message ===" -ForegroundColor Cyan
}

function Get-PythonCommand {
    $candidates = @(
        @{ Cmd = "py"; Args = @("-3", "--version") },
        @{ Cmd = "python"; Args = @("--version") },
        @{ Cmd = "python3"; Args = @("--version") }
    )

    foreach ($candidate in $candidates) {
        try {
            $null = & $candidate.Cmd @($candidate.Args) 2>$null
            if ($LASTEXITCODE -eq 0) {
                return $candidate.Cmd
            }
        }
        catch {
        }
    }

    throw "Python 3 was not found. Install Python 3 and make sure it is in PATH."
}

function Invoke-Python {
    param(
        [Parameter(Mandatory = $true)][string]$PythonCmd,
        [Parameter(Mandatory = $true)][string[]]$Args
    )

    if ($PythonCmd -eq "py") {
        & py -3 @Args
    }
    else {
        & $PythonCmd @Args
    }

    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed: $PythonCmd $($Args -join ' ')"
    }
}

Write-Step "Preparing build environment"
$scriptDir = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($scriptDir)) {
    $scriptDir = (Get-Location).Path
}
if ([string]::IsNullOrWhiteSpace($scriptDir)) {
    throw "Could not determine the script directory."
}
Set-Location -Path $scriptDir
Write-Host "Working directory: $scriptDir"

$pythonCmd = Get-PythonCommand
Write-Host "Using Python launcher: $pythonCmd"

$venvPath = Join-Path $scriptDir $VenvDir
$venvPython = Join-Path $venvPath "Scripts\python.exe"
$venvPip = Join-Path $venvPath "Scripts\pip.exe"
$requirementsPath = Join-Path $scriptDir $RequirementsFile
$specPath = Join-Path $scriptDir $SpecFile
$entryPath = Join-Path $scriptDir $EntryScript
$distDir = Join-Path $scriptDir "dist"
$buildDir = Join-Path $scriptDir "build"

if ($Clean) {
    Write-Step "Cleaning old build artifacts"
    foreach ($target in @($distDir, $buildDir)) {
        if (Test-Path $target) {
            Remove-Item -Recurse -Force $target
            Write-Host "Removed $target"
        }
    }
}

if (-not (Test-Path $venvPython)) {
    Write-Step "Creating virtual environment"
    Invoke-Python -PythonCmd $pythonCmd -Args @("-m", "venv", $venvPath)
}
else {
    Write-Step "Reusing existing virtual environment"
}

Write-Step "Bootstrapping pip"
& $venvPython -m ensurepip --upgrade
if ($LASTEXITCODE -ne 0) {
    throw "Failed to bootstrap pip inside the virtual environment."
}

Write-Step "Upgrading pip, setuptools, and wheel"
& $venvPython -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) {
    throw "Failed to upgrade pip/setuptools/wheel."
}

if (-not $SkipRequirements) {
    if (-not (Test-Path $requirementsPath)) {
        throw "Requirements file not found: $requirementsPath"
    }

    Write-Step "Installing requirements from $RequirementsFile"
    & $venvPython -m pip install -r $requirementsPath
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install dependencies from $RequirementsFile"
    }
}
else {
    Write-Step "Skipping dependency installation"
}

Write-Step "Verifying project files"
if ($UseSpec) {
    if (-not (Test-Path $specPath)) {
        throw "Spec file not found: $specPath"
    }
    Write-Host "Using spec file: $SpecFile"
}
else {
    if (-not (Test-Path $entryPath)) {
        throw "Entry script not found: $entryPath"
    }
    Write-Host "Using entry script: $EntryScript"
}

Write-Step "Building executable"
if ($UseSpec) {
    & $venvPython -m PyInstaller --clean $specPath
}
else {
    & $venvPython -m PyInstaller --clean --noconfirm --onefile --windowed --name $ExeName $entryPath
}

if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller build failed."
}

$exePath = Join-Path $distDir "$ExeName.exe"
Write-Step "Build complete"
if (Test-Path $exePath) {
    Write-Host "Executable created successfully:" -ForegroundColor Green
    Write-Host $exePath -ForegroundColor Green
}
else {
    Write-Warning "Build finished, but the expected executable was not found at: $exePath"
    if (Test-Path $distDir) {
        Write-Host "Contents of dist folder:"
        Get-ChildItem $distDir | Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize
    }
}

Write-Host ""
Write-Host "To activate the build environment later, run:" -ForegroundColor Yellow
Write-Host "$VenvDir\Scripts\Activate.ps1"
