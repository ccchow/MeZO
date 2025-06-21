#!/bin/bash
# MeZO Paper Reproduction Setup Script
# This script helps set up the environment for reproducing MeZO paper results

set -e  # Exit on any error

echo "🚀 MeZO Paper Reproduction Setup"
echo "=================================="

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "📍 Python version: $python_version"

if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)'; then
    echo "❌ Python 3.8+ required, found $python_version"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "accelerate_mezo.py" ]; then
    echo "❌ Please run this script from the MeZO project directory"
    echo "   Make sure accelerate_mezo.py is in the current directory"
    exit 1
fi

echo "✅ Environment check passed"

# Install/upgrade key packages
echo ""
echo "📦 Installing/upgrading packages..."

# Core ML packages
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install --upgrade transformers>=4.49.0
pip install --upgrade accelerate>=0.28.0
pip install --upgrade datasets>=2.19.0
pip install --upgrade peft>=0.4.0

# Analysis and utilities
pip install --upgrade pandas matplotlib seaborn
pip install --upgrade tqdm numpy scipy
pip install --upgrade safetensors

# Optional but recommended
pip install --upgrade bitsandbytes  # For 8-bit optimization
pip install --upgrade wandb         # For experiment tracking

echo "✅ Package installation complete"

# Check CUDA availability
echo ""
echo "🔍 Checking CUDA setup..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✅ CUDA available: {torch.cuda.device_count()} GPUs')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1024**3
        print(f'   GPU {i}: {props.name} ({memory_gb:.1f} GB)')
else:
    print('⚠️  CUDA not available - will run on CPU (very slow)')
"

# Set up environment variables
echo ""
echo "⚙️  Setting up environment..."

# Create environment script
cat > setup_env.sh << 'EOF'
#!/bin/bash
# Source this file to set up MeZO environment
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo "✅ MeZO environment variables set"
EOF

chmod +x setup_env.sh
echo "✅ Created setup_env.sh (source this before running experiments)"

# Create directories for results
mkdir -p results
mkdir -p results/roberta_experiments
mkdir -p results/opt_experiments
mkdir -p results/analysis
echo "✅ Created results directories"

# Run validation test
echo ""
echo "🧪 Running validation test..."
if python3 test_reproduction_setup.py; then
    echo ""
    echo "🎉 Setup complete! You're ready to reproduce MeZO paper results."
    echo ""
    echo "📋 Quick start commands:"
    echo ""
    echo "# Source environment (run this in each new terminal)"
    echo "source setup_env.sh"
    echo ""
    echo "# Test with small model (recommended first step)"
    echo "python accelerate_mezo.py \\"
    echo "    --model_name distilgpt2 \\"
    echo "    --dataset imdb \\"
    echo "    --text_column text \\"
    echo "    --batch_size 4 \\"
    echo "    --max_steps 50 \\"
    echo "    --memory_logging \\"
    echo "    --output_dir ./results/test_run"
    echo ""
    echo "# RoBERTa-large paper reproduction"
    echo "python run_mezo_paper_experiments.py \\"
    echo "    --model_family roberta \\"
    echo "    --model_name roberta-large \\"
    echo "    --task sst2 \\"
    echo "    --output_dir ./results/roberta_experiments"
    echo ""
    echo "# Full grid search (warning: takes many hours)"
    echo "python run_mezo_paper_experiments.py \\"
    echo "    --model_family roberta \\"
    echo "    --model_name roberta-large \\"
    echo "    --grid_search \\"
    echo "    --output_dir ./results/roberta_experiments"
    echo ""
    echo "📊 For analysis:"
    echo "python analyze_mezo_results.py \\"
    echo "    --results_dir ./results/roberta_experiments \\"
    echo "    --create_plots \\"
    echo "    --output_dir ./results/analysis"
    echo ""
    echo "📚 See REPRODUCTION_GUIDE.md for detailed instructions"
else
    echo ""
    echo "❌ Validation test failed. Please check the errors above."
    echo "   Common issues:"
    echo "   - Missing dependencies (install with pip)"
    echo "   - CUDA not properly set up"
    echo "   - Insufficient GPU memory"
    echo ""
    echo "   Try the test script directly:"
    echo "   python test_reproduction_setup.py"
fi

echo ""
echo "=================================="
echo "🏁 Setup script completed"
