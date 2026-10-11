# Clone the model repos (pinned commits) and download the checkpoints.
#   powershell -ExecutionPolicy Bypass -File scripts/setup_models.ps1
# Safe to run again: existing folders/files are skipped.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
New-Item -ItemType Directory -Force "$root\third_party", "$root\checkpoints" | Out-Null

$repos = @(
    @{ name = "sam2";                 url = "https://github.com/facebookresearch/sam2.git";            commit = "2b90b9f5ceec907a1c18123530e92e794ad901a4" },
    @{ name = "MatAnyone";            url = "https://github.com/pq-yang/MatAnyone.git";                commit = "e5ddc534c1fff9bb9e54cf476095d29071b7cb4f" },
    @{ name = "Video-Depth-Anything"; url = "https://github.com/DepthAnything/Video-Depth-Anything.git"; commit = "4f5ae23172ba60fd7bc11ef671cca678842c7072" },
    @{ name = "Depth-Anything-V2";    url = "https://github.com/DepthAnything/Depth-Anything-V2.git";   commit = "a561b849ebae10a6f5ef49e26c83cbbcd36c71bf" }
)
foreach ($r in $repos) {
    $dst = "$root\third_party\$($r.name)"
    if (-not (Test-Path $dst)) {
        git clone -q $r.url $dst
        git -C $dst checkout -q $r.commit
    }
    Write-Host "repo ok: $($r.name)"
}

$ckpts = @{
    "sam2.1_hiera_small.pt"         = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
    "matanyone.pth"                 = "https://github.com/pq-yang/MatAnyone/releases/download/v1.0.0/matanyone.pth"
    "video_depth_anything_vits.pth" = "https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth"
    "depth_anything_v2_vits.pth"    = "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth"
}
foreach ($k in $ckpts.Keys) {
    $dst = "$root\checkpoints\$k"
    if (-not (Test-Path $dst)) { curl.exe -sL -o $dst $ckpts[$k] }
    Write-Host ("checkpoint ok: {0} ({1:N0} MB)" -f $k, ((Get-Item $dst).Length / 1MB))
}
