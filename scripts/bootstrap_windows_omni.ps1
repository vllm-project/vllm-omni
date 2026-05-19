# SPDX-License-Identifier: Apache-2.0
<#
.SYNOPSIS
Build and prove the Windows vLLM + vLLM-Omni Qwen3-Omni stack from scratch.

.DESCRIPTION
This script bootstraps the Windows vLLM source build, installs vLLM-Omni from
source, downloads the Qwen3-Omni model through Hugging Face, and runs the
Windows audio in -> audio out streaming probe. The probe records metadata only
and does not write an output WAV file.
#>

[CmdletBinding()]
param(
    [string]$InstallRoot = "C:\tmp\vllm-omni-windows-bootstrap",
    [string]$VllmRepoUrl = "https://github.com/ericleigh007/vllm-windows.git",
    [string]$OmniRepoUrl = "https://github.com/ericleigh007/vllm-omni-windows.git",
    [string]$Branch = "windows-compat",
    [string]$VllmRepoPath = "",
    [string]$OmniRepoPath = "",
    [string]$VenvPath = "C:\tmp\vllmvenv",
    [string]$CudaPath = "C:\tmp\cuda13",
    [string]$CudaArch = "120",
    [string]$FetchContentBaseDir = "C:\tmp\vllm_deps",
    [int]$MaxJobs = 4,
    [int]$NvccThreads = 1,
    [string]$ModelId = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    [string]$ModelLocalDir = "",
    [string]$ProofOut = "",
    [switch]$SkipVllmBootstrap,
    [switch]$SkipDependencyInstall,
    [switch]$SkipModelDownload,
    [switch]$SkipUnitTests,
    [switch]$SkipProof,
    [switch]$AllowDirtyRepo
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Require-Command {
    param([string]$Name, [string]$InstallHint)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "$Name was not found on PATH. $InstallHint"
    }
}

function Invoke-Native {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments
    )
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $($Arguments -join ' ')"
    }
}

function Ensure-Repo {
    param([string]$Path, [string]$Url, [string]$Ref)
    if (-not (Test-Path $Path)) {
        Write-Step "Cloning $Url ($Ref) to $Path"
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Path) | Out-Null
        Invoke-Native "git.exe" "clone" "-b" $Ref $Url $Path
        return
    }

    if (-not (Test-Path (Join-Path $Path ".git"))) {
        throw "$Path exists but is not a Git repository."
    }

    Push-Location $Path
    try {
        $dirty = git status --porcelain
        if ($dirty -and -not $AllowDirtyRepo) {
            throw "$Path has uncommitted changes. Re-run with -AllowDirtyRepo to skip the safety check."
        }
        Write-Step "Updating existing repository at $Path"
        Invoke-Native "git.exe" "fetch" "origin" $Ref
        Invoke-Native "git.exe" "checkout" $Ref
        Invoke-Native "git.exe" "pull" "--ff-only" "origin" $Ref
    }
    finally {
        Pop-Location
    }
}

function Get-PythonExe {
    param([string]$Path)
    $python = Join-Path $Path "Scripts\python.exe"
    if (-not (Test-Path $python)) {
        throw "Python virtualenv was not found at $Path. Run without -SkipVllmBootstrap first."
    }
    return $python
}

function Resolve-CudaToolkitPath {
    param([string]$PreferredPath)
    $candidates = @(
        $PreferredPath,
        "C:\tmp\cuda13_system",
        $env:CUDA_PATH,
        $env:CUDA_HOME,
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
    ) | Where-Object { $_ } | Select-Object -Unique

    foreach ($candidate in $candidates) {
        if (Test-Path (Join-Path $candidate "bin\nvcc.exe")) {
            return $candidate
        }
    }
    throw "Could not find a CUDA Toolkit with bin\nvcc.exe. Pass -CudaPath or create a space-free CUDA junction."
}

function Set-OmniEnvironment {
    param([string]$PythonExe)
    $script:ResolvedCudaPath = Resolve-CudaToolkitPath -PreferredPath $CudaPath
    $script:CudaPath = $script:ResolvedCudaPath
    $torchLib = Join-Path (Split-Path -Parent (Split-Path -Parent $PythonExe)) "Lib\site-packages\torch\lib"
    $venvScripts = Split-Path -Parent $PythonExe
    $env:PATH = "$venvScripts;$CudaPath\bin;$torchLib;$env:PATH"
    $env:CUDA_HOME = $CudaPath
    $env:CUDA_PATH = $CudaPath
    $env:PYTHONPATH = $OmniRepoPath
    $env:HF_HUB_DISABLE_SYMLINKS = "1"
    $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
    $env:VLLM_WORKER_MULTIPROC_METHOD = "spawn"
    $env:VLLM_USE_FLASHINFER_SAMPLER = "0"
    $env:VLLM_HOST_IP = "127.0.0.1"
}

function Download-HfSnapshot {
    param([string]$PythonExe, [string]$RepoId, [string]$LocalDir)
    $code = @"
from huggingface_hub import snapshot_download

kwargs = {
    "repo_id": r"$RepoId",
    "resume_download": True,
}
if r"$LocalDir":
    kwargs["local_dir"] = r"$LocalDir"
snapshot_download(**kwargs)
print("Downloaded", r"$RepoId")
"@
    Invoke-Native $PythonExe "-c" $code
}

Require-Command "git.exe" "Install Git for Windows."

if (-not $VllmRepoPath) {
    $VllmRepoPath = Join-Path $InstallRoot "vllm-windows"
}
if (-not $OmniRepoPath) {
    $scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
    $candidateRepo = Resolve-Path (Join-Path $scriptRoot "..") -ErrorAction SilentlyContinue
    if ($candidateRepo -and (Test-Path (Join-Path $candidateRepo ".git"))) {
        $OmniRepoPath = $candidateRepo.Path
    }
    else {
        $OmniRepoPath = Join-Path $InstallRoot "vllm-omni-windows"
    }
}
if (-not $ProofOut) {
    $ProofOut = Join-Path $OmniRepoPath "demo_outputs\bootstrap_qwen3_omni_audio_stream_probe.json"
}

if (-not $SkipVllmBootstrap) {
    Ensure-Repo -Path $VllmRepoPath -Url $VllmRepoUrl -Ref $Branch
    $vllmBootstrap = Join-Path $VllmRepoPath "scripts\bootstrap_windows.ps1"
    if (-not (Test-Path $vllmBootstrap)) {
        throw "vLLM bootstrap script was not found at $vllmBootstrap."
    }

    Write-Step "Bootstrapping vLLM Windows source build"
    $vllmArgs = @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $vllmBootstrap,
        "-RepoPath", $VllmRepoPath,
        "-VenvPath", $VenvPath,
        "-CudaPath", $CudaPath,
        "-CudaArch", $CudaArch,
        "-FetchContentBaseDir", $FetchContentBaseDir,
        "-MaxJobs", [string]$MaxJobs,
        "-NvccThreads", [string]$NvccThreads
    )
    if ($AllowDirtyRepo) {
        $vllmArgs += "-AllowDirtyRepo"
    }
    Invoke-Native "powershell.exe" @vllmArgs
}

Ensure-Repo -Path $OmniRepoPath -Url $OmniRepoUrl -Ref $Branch
$pythonExe = Get-PythonExe -Path $VenvPath
Set-OmniEnvironment -PythonExe $pythonExe

Push-Location $OmniRepoPath
try {
    if (-not $SkipDependencyInstall) {
        Write-Step "Installing vLLM-Omni runtime dependencies"
        Invoke-Native $pythonExe "-m" "pip" "install" "-r" "requirements\common.txt" "onnxruntime>=1.23.2" "pytest>=8.0.0"
    }

    Write-Step "Installing vLLM-Omni editable"
    Invoke-Native $pythonExe "-m" "pip" "install" "-e" "." "--no-build-isolation" "--no-deps"

    if (-not $SkipUnitTests) {
        Write-Step "Running focused Windows async streaming tests"
        Invoke-Native $pythonExe "-m" "pytest" "tests\distributed\omni_connectors\test_chunk_transfer_adapter.py" "tests\engine\test_async_omni_engine_outputs.py"
    }

    if (-not $SkipModelDownload) {
        Write-Step "Downloading Qwen3-Omni model snapshot"
        Download-HfSnapshot -PythonExe $pythonExe -RepoId $ModelId -LocalDir $ModelLocalDir
    }

    if (-not $SkipProof) {
        Write-Step "Running Qwen3-Omni audio in -> audio out streaming proof"
        Invoke-Native $pythonExe "examples\offline_inference\qwen3_omni\windows_audio_stream_probe.py" "--model" $ModelId "--deploy-config" "vllm_omni\deploy\qwen3_omni_moe_windows_single_gpu.yaml" "--stop-after-audio-chunks" "1" "--out" $ProofOut
    }
}
finally {
    Pop-Location
}

Write-Step "vLLM-Omni Windows bootstrap complete"
Write-Host "vLLM repo:     $VllmRepoPath"
Write-Host "Omni repo:     $OmniRepoPath"
Write-Host "Virtualenv:    $VenvPath"
Write-Host "CUDA:          $CudaPath"
Write-Host "Proof output:  $ProofOut"
