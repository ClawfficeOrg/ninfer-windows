# NInfer Windows build environment for the RTX 5070 Ti (sm_120a) port.
# Dot-source this, or let build.ps1 import it.
#
# Selects CUDA 13.1 (the validated toolkit; 13.0 is also on PATH by default and
# the repo README pins >= 13.1), and locates vcpkg shipped inside the VS install
# because there is no standalone C:\src\vcpkg on this machine.

$cudaRoot = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1'

if (-not (Test-Path (Join-Path $cudaRoot 'bin\nvcc.exe'))) {
    throw "CUDA 13.1 not found at $cudaRoot"
}

$env:CUDA_PATH = $cudaRoot
$env:CUDAToolkit_ROOT = $cudaRoot
$env:Path = (Join-Path $cudaRoot 'bin') + ';' + $env:Path

$vsInstallCandidates = @(
    'C:\Program Files\Microsoft Visual Studio\2022\BuildTools'
    'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools'
    'C:\Program Files\Microsoft Visual Studio\17\Community'
    'C:\Program Files\Microsoft Visual Studio\18\Community'
)

$vsRoot = $vsInstallCandidates | Where-Object { Test-Path (Join-Path $_ 'VC\Auxiliary\Build\vcvars64.bat') } | Select-Object -First 1

if (-not $vsRoot) {
    throw "No Visual Studio install with VC\Auxiliary\Build\vcvars64.bat found in: $($vsInstallCandidates -join ', ')"
}
$env:NINFER_VS_ROOT = $vsRoot

$vcpkgCandidates = @(
    (Join-Path $vsRoot 'VC\vcpkg')
    'C:\src\vcpkg'
    'C:\Program Files\Microsoft Visual Studio\18\Community\VC\vcpkg'
)

$vcpkgRoot = $vcpkgCandidates | Where-Object {
    Test-Path (Join-Path $_ 'scripts\buildsystems\vcpkg.cmake')
} | Select-Object -First 1

if (-not $vcpkgRoot) {
    throw "vcpkg (with scripts\buildsystems\vcpkg.cmake) not found in: $($vcpkgCandidates -join ', ')"
}

$env:VCPKG_ROOT = $vcpkgRoot
$env:VCPKG_TOOLCHAIN_FILE = Join-Path $vcpkgRoot 'scripts\buildsystems\vcpkg.cmake'

Write-Host "CUDA_PATH            = $env:CUDA_PATH"
Write-Host "VCPKG_ROOT           = $env:VCPKG_ROOT"
Write-Host "VS root              = $env:NINFER_VS_ROOT"
Write-Host "nvcc                 = $((Get-Command nvcc).Source)"
