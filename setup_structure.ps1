# Run from the root of your cloned TextSus repo:
#   .\setup_structure.ps1
# If scripts are blocked:
#   powershell -ExecutionPolicy Bypass -File .\setup_structure.ps1

$ErrorActionPreference = "Stop"

function Touch($path) {
    $dir = Split-Path $path -Parent
    if ($dir -and -not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
    if (-not (Test-Path $path)) {
        New-Item -ItemType File -Path $path -Force | Out-Null
    }
}

# Package __init__ files
Touch "src/textsus/__init__.py"
foreach ($p in "seed","gvalues","sampling","scoring","generation","detection","evaluation","utils") {
    Touch "src/textsus/$p/__init__.py"
}

# Stub files
$files = @(
    # Member 2: core method
    "src/textsus/seed/random_seed.py",
    "src/textsus/gvalues/gvalues.py",
    "src/textsus/sampling/tournament.py",
    "src/textsus/sampling/gumbel.py",
    "src/textsus/sampling/soft_red_list.py",
    "src/textsus/sampling/context_masking.py",
    "src/textsus/scoring/mean_score.py",
    "src/textsus/scoring/weighted_mean.py",
    "src/textsus/detection/detector.py",
    # Member 3: generation and evaluation
    "src/textsus/generation/watermarked_generator.py",
    "src/textsus/evaluation/metrics.py",
    "src/textsus/evaluation/perplexity.py",
    "src/textsus/evaluation/self_bleu.py",
    "src/textsus/evaluation/latency.py",
    # Configs, experiments
    "configs/default.yaml",
    "experiments/run_detectability.py",
    "experiments/run_quality.py",
    "experiments/run_distortionary.py",
    "experiments/run_latency.py",
    # Demo and tests
    "demo/app.py",
    "tests/test_seed.py",
    "tests/test_gvalues.py",
    "tests/test_tournament.py",
    "tests/test_scoring.py",
    # Keep empty folders in git
    "configs/experiments/.gitkeep",
    "data/.gitkeep",
    "results/figures/.gitkeep",
    "results/tables/.gitkeep",
    "notebooks/.gitkeep",
    "docs/01_background/.gitkeep",
    "docs/02_method/.gitkeep",
    "docs/03_evaluation/.gitkeep",
    "paper/.gitkeep",
    "report/.gitkeep",
    "report/slides/.gitkeep"
)
foreach ($f in $files) { Touch $f }

Set-Content -Path "data/README.md" -Value "Download and split instructions for ELI5 prompts go here."

# requirements.txt (only if missing)
if (-not (Test-Path "requirements.txt")) {
@'
torch
transformers
datasets
numpy
scipy
pandas
matplotlib
pyyaml
tqdm
nltk
pytest
streamlit
'@ | Set-Content -Path "requirements.txt"
}

# .gitignore (only if missing)
if (-not (Test-Path ".gitignore")) {
@'
__pycache__/
*.pyc
.venv/
.env
.ipynb_checkpoints/
.DS_Store
data/*
!data/.gitkeep
!data/README.md
*.pt
*.bin
*.safetensors
'@ | Set-Content -Path ".gitignore"
}

Write-Host "TextSus scaffold created."