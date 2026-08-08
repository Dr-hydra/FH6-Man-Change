[CmdletBinding()]
param(
    [Parameter()]
    [string]$ConfigPath,

    [Parameter()]
    [switch]$ReuseBaseline,

    [Parameter()]
    [switch]$AllLods,

    [Parameter()]
    [switch]$ReuseComponents
)

$workspaceRoot = Split-Path -Parent $PSScriptRoot
$blenderPath = 'D:\SteamLibrary\steamapps\common\Blender\blender.exe'
$probeRunner = 'C:\Users\28377\.codex\skills\fh6-blender-pipeline\scripts\run_blender_probe.ps1'
if (-not $ConfigPath) {
    $ConfigPath = Join-Path $workspaceRoot 'sources\si\source.config.json'
}
if (-not (Test-Path -LiteralPath $blenderPath)) {
    throw "Blender not found: $blenderPath"
}
if (-not (Test-Path -LiteralPath $ConfigPath)) {
    throw "Source config not found: $ConfigPath"
}
if (-not (Test-Path -LiteralPath $probeRunner)) {
    throw "Probe runner not found: $probeRunner"
}

$config = Get-Content -Raw -LiteralPath $ConfigPath | ConvertFrom-Json
if ($config.primary_format -ne 'fbx') {
    throw "The primary source format must be fbx"
}

function Resolve-WorkspacePath([string]$RelativePath) {
    return [System.IO.Path]::GetFullPath((Join-Path $workspaceRoot $RelativePath))
}

$sourceFbx = Resolve-WorkspacePath $config.primary.path
$baselineBlend = Resolve-WorkspacePath $config.outputs.baseline_blend
$baselineMetadata = Resolve-WorkspacePath $config.outputs.baseline_metadata
$baselineProbe = Resolve-WorkspacePath $config.outputs.baseline_probe
$importScript = Join-Path $PSScriptRoot 'import_fbx_baseline.py'
$componentScript = Join-Path $PSScriptRoot 'create_fbx_component_split.py'
$globalScale = [double]$config.primary.global_scale
$workingLod = [string]$config.primary.working_lod

foreach ($path in @($sourceFbx, $importScript, $componentScript)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required input not found: $path"
    }
}
$baselinePaths = @($baselineBlend, $baselineMetadata, $baselineProbe)
$existingBaselinePaths = @($baselinePaths | Where-Object { Test-Path -LiteralPath $_ })
if ($existingBaselinePaths.Count -gt 0) {
    if ($existingBaselinePaths.Count -ne $baselinePaths.Count) {
        throw "The existing baseline is incomplete; refusing to continue"
    }
    if (-not $ReuseBaseline) {
        throw "Baseline already exists. Pass -ReuseBaseline to build a new component milestone from it."
    }
    $existingMetadata = Get-Content -Raw -LiteralPath $baselineMetadata | ConvertFrom-Json
    $currentSourceHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $sourceFbx).Hash.ToLowerInvariant()
    if ($existingMetadata.source.fbx_sha256.ToLowerInvariant() -ne $currentSourceHash) {
        throw "Existing baseline source hash does not match the configured FBX"
    }
    if ([int]$existingMetadata.validation.hard_error_count -ne 0) {
        throw "Existing baseline report contains hard errors"
    }
} else {
    & $blenderPath --background --factory-startup --python $importScript -- `
        --fbx $sourceFbx `
        --blend $baselineBlend `
        --metadata $baselineMetadata `
        --global-scale $globalScale
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    $baselineProbeOutput = & $probeRunner -BlendFile $baselineBlend -OutputJson $baselineProbe -FailOnInvalid 2>&1
    $baselineProbeExitCode = $LASTEXITCODE
    $baselineProbeOutput | Where-Object { $_ -notmatch '^FH6_PROBE_JSON=' }
    if ($baselineProbeExitCode -ne 0) {
        exit $baselineProbeExitCode
    }
}

$componentTargets = @()
if ($AllLods) {
    foreach ($lod in @($config.primary.working_lods)) {
        $entryProperty = $config.outputs.components.PSObject.Properties[[string]$lod]
        if (-not $entryProperty) {
            throw "No component output configuration found for $lod"
        }
        $entry = $entryProperty.Value
        $componentTargets += [pscustomobject]@{
            Lod = [string]$lod
            Blend = Resolve-WorkspacePath $entry.blend
            Report = Resolve-WorkspacePath $entry.report
            Probe = Resolve-WorkspacePath $entry.probe
        }
    }
} else {
    $componentTargets += [pscustomobject]@{
        Lod = $workingLod
        Blend = Resolve-WorkspacePath $config.outputs.component_blend
        Report = Resolve-WorkspacePath $config.outputs.component_report
        Probe = Resolve-WorkspacePath $config.outputs.component_probe
    }
}

foreach ($target in $componentTargets) {
    $targetPaths = @($target.Blend, $target.Report, $target.Probe)
    $existingTargetPaths = @($targetPaths | Where-Object { Test-Path -LiteralPath $_ })
    if ($existingTargetPaths.Count -gt 0) {
        if ($existingTargetPaths.Count -ne $targetPaths.Count) {
            throw "The existing $($target.Lod) component milestone is incomplete; refusing to continue"
        }
        if (-not $ReuseComponents) {
            throw "Component milestone already exists for $($target.Lod). Pass -ReuseComponents to preserve and verify it."
        }
        $existingComponentReport = Get-Content -Raw -LiteralPath $target.Report | ConvertFrom-Json
        $existingComponentProbe = Get-Content -Raw -LiteralPath $target.Probe | ConvertFrom-Json
        if ([int]$existingComponentReport.validation.hard_error_count -ne 0) {
            throw "Existing $($target.Lod) component report contains hard errors"
        }
        if ([int]$existingComponentProbe.summary.hard_error_count -ne 0) {
            throw "Existing $($target.Lod) component probe contains hard errors"
        }
        Write-Output "FH6_FBX_COMPONENTS_REUSED=$($target.Lod):$($target.Blend)"
        continue
    }

    & $blenderPath --background --factory-startup --python $componentScript -- `
        --source-blend $baselineBlend `
        --blend $target.Blend `
        --report $target.Report `
        --lod $target.Lod
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    $componentProbeOutput = & $probeRunner -BlendFile $target.Blend -OutputJson $target.Probe -FailOnInvalid 2>&1
    $componentProbeExitCode = $LASTEXITCODE
    $componentProbeOutput | Where-Object { $_ -notmatch '^FH6_PROBE_JSON=' }
    if ($componentProbeExitCode -ne 0) {
        exit $componentProbeExitCode
    }
    Write-Output "FH6_FBX_COMPONENTS_BUILT=$($target.Lod):$($target.Blend)"
}

Write-Output "FH6_FBX_SOURCE=$sourceFbx"
Write-Output "FH6_FBX_BASELINE=$baselineBlend"
foreach ($target in $componentTargets) {
    Write-Output "FH6_FBX_COMPONENTS=$($target.Lod):$($target.Blend)"
}
