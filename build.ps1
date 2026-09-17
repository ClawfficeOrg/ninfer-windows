# Configure + build NInfer for Windows in the VS 2022 (MSVC) environment.
#
#   .\build.ps1              # configure if needed, then build Release
#   .\build.ps1 -Reconfigure # wipe build-windows and configure fresh
#
# Must run under cmd's vcvars64 so cl.exe / MSVC headers resolve; this script
# re-invokes itself through vcvars64.bat automatically.

param(
    [switch]$Reconfigure,
    [switch]$ConfigureOnly,
    [string]$BuildDir = 'build-windows'
)

$ErrorActionPreference = 'Stop'

if (-not $env:NINFER_IN_VCVARS) {
    . (Join-Path $PSScriptRoot 'env-build.ps1')

    $vcvars = Join-Path $env:NINFER_VS_ROOT 'VC\Auxiliary\Build\vcvars64.bat'
    Write-Host "Entering MSVC environment via $vcvars"

    $innerArgs = @()
    if ($Reconfigure) { $innerArgs += '-Reconfigure' }
    if ($ConfigureOnly) { $innerArgs += '-ConfigureOnly' }
    if ($BuildDir -ne 'build-windows') { $innerArgs += @('-BuildDir', $BuildDir) }

    $cmdLine = "set NINFER_IN_VCVARS=1 && powershell -NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" $($innerArgs -join ' ')"
    & cmd /c "`"$vcvars`" >nul 2>&1 && $cmdLine"
    exit $LASTEXITCODE
}

. (Join-Path $PSScriptRoot 'env-build.ps1')

$buildPath = Join-Path $PSScriptRoot $BuildDir

if ($Reconfigure -and (Test-Path $buildPath)) {
    Write-Host "Removing $buildPath"
    Remove-Item -Recurse -Force $buildPath
}

if (-not (Test-Path (Join-Path $buildPath 'CMakeCache.txt'))) {
    Write-Host "Configuring in $buildPath"
    cmake -S $PSScriptRoot -B $buildPath `
        -G "Visual Studio 17 2022" -A x64 `
        -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_TOOLCHAIN_FILE" `
        -DVCPKG_TARGET_TRIPLET=x64-windows `
        -DCUDAToolkit_ROOT="$env:CUDAToolkit_ROOT" `
        -DCMAKE_CUDA_COMPILER="$(Join-Path $env:CUDA_PATH 'bin\nvcc.exe')" `
        -DCMAKE_CUDA_ARCHITECTURES=120a
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} else {
    Write-Host "Reusing existing configuration in $buildPath (pass -Reconfigure to wipe)"
}

if ($ConfigureOnly) { exit 0 }

Write-Host "Building Release"
cmake --build $buildPath --config Release --parallel
exit $LASTEXITCODE
