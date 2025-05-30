#!/bin/bash

# Setup script for Resource Allocation Simulation Framework

set -e  # Exit on any error

echo "🚀 Setting up Resource Allocation Simulation Framework..."

# Parse command line arguments
INSTALL_DEV=false
RUN_TESTS=false
FORCE_REINSTALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dev)
            INSTALL_DEV=true
            shift
            ;;
        --test)
            RUN_TESTS=true
            shift
            ;;
        --force)
            FORCE_REINSTALL=true
            shift
            ;;
        *)
            echo "Unknown option $1"
            echo "Usage: $0 [--dev] [--test] [--force]"
            echo "  --dev    Install development dependencies"
            echo "  --test   Run tests after setup"
            echo "  --force  Force reinstall of all dependencies"
            exit 1
            ;;
    esac
done

# Check Python version
echo "Checking Python version..."
python_version=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
echo "Python version: $python_version"

if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "✅ Python version is compatible"
else
    echo "❌ Python 3.8+ required. Current version: $python_version"
    exit 1
fi

# Create virtual environment if it doesn't exist or force reinstall
if [ ! -d "venv" ] || [ "$FORCE_REINSTALL" = true ]; then
    if [ "$FORCE_REINSTALL" = true ] && [ -d "venv" ]; then
        echo "Removing existing virtual environment..."
        rm -rf venv
    fi
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Verify virtual environment is active
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment activated: $VIRTUAL_ENV"
else
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

# Upgrade pip, setuptools, and wheel
echo "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel

# Install core dependencies explicitly first
echo "Installing core dependencies..."
pip install numpy>=1.20.0 matplotlib>=3.5.0 pandas>=1.3.0 click>=8.0.0 pyyaml>=6.0 seaborn>=0.11.0 scipy>=1.7.0

# Install package in development mode
echo "Installing package in development mode..."
pip install -e .

# Verify core installation
echo "Verifying core package installation..."
python3 -c "
import resource_allocation_sim
print(f'✅ Core package version: {resource_allocation_sim.__version__}')
print(f'✅ Author: {resource_allocation_sim.__author__}')
"

# Install optional dependencies
echo "Installing optional dependencies..."
pip install -e ".[full]" || {
    echo "⚠️  Some optional dependencies failed to install"
    echo "Installing individual optional packages..."
    
    # Try installing optional dependencies individually
    echo "Installing networkx for network visualisations..."
    pip install "networkx>=2.6" || echo "⚠️  networkx installation failed"
    
    echo "Installing plotly for interactive plots..."
    pip install "plotly>=5.0.0" || echo "⚠️  plotly installation failed"
    
    echo "Installing jupyter and ipywidgets..."
    pip install "jupyter>=1.0.0" "ipywidgets>=7.6.0" || echo "⚠️  jupyter installation failed"
    
    echo "Installing mpltern for ternary plots..."
    pip install "mpltern>=1.0.0" || echo "⚠️  mpltern installation failed (this is optional)"
}

# Install dev dependencies if requested
if [ "$INSTALL_DEV" = true ]; then
    echo "Installing development dependencies..."
    pip install -e ".[dev]" || {
        echo "Installing individual dev packages..."
        pip install pytest>=6.0 pytest-cov>=3.0 black>=22.0 isort>=5.0 mypy>=0.950 pre-commit>=2.15 flake8>=4.0
    }
fi

# List installed packages
echo "📦 Installed packages:"
pip list | grep -E "(resource-allocation-sim|numpy|pandas|matplotlib|scipy|seaborn|click|pyyaml)"

# Check for optional packages
echo ""
echo "🔍 Checking optional dependencies:"
python3 -c "
try:
    import networkx
    print('✅ networkx available for network visualisations')
except ImportError:
    print('❌ networkx not available (network visualisations disabled)')

try:
    import mpltern
    print('✅ mpltern available for ternary plots')
except ImportError:
    print('❌ mpltern not available (ternary plots disabled)')

try:
    import plotly
    print('✅ plotly available for interactive plots')
except ImportError:
    print('❌ plotly not available (interactive plots disabled)')
"

# Create a simple test if it doesn't exist
if [ ! -f "run_simple_test.py" ]; then
    echo "Creating simple test file..."
    cat > run_simple_test.py << 'EOF'
#!/usr/bin/env python3
"""Simple test to verify installation."""

try:
    # Test proper package imports
    from resource_allocation_sim.core.simulation import SimulationRunner
    from resource_allocation_sim.utils.config import Config
    
    print("✅ Import test passed!")
    
    # Run a minimal simulation with 2 resources
    config = Config()
    config.num_iterations = 10
    config.num_agents = 3
    config.num_resources = 2
    config.capacity = [1.0, 1.0]  # Match the number of resources
    
    runner = SimulationRunner(config)
    runner.setup()
    results = runner.run()
    
    print(f"✅ Simulation test passed! Final consumption: {results['final_consumption']}")
    print("🎉 Installation successful!")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Check package structure and installation")
    exit(1)
except Exception as e:
    print(f"❌ Test failed: {e}")
    exit(1)
EOF
fi

# Run simple test
echo ""
echo "🧪 Running installation verification test..."
python3 run_simple_test.py

# Run proper tests if requested and pytest is available
if [ "$RUN_TESTS" = true ]; then
    if command -v pytest &> /dev/null; then
        echo ""
        echo "🧪 Running pytest..."
        pytest tests/ -v
    else
        echo "⚠️  pytest not available. Install with: pip install -e \".[dev]\""
    fi
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "🎯 To use the framework:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run CLI help: resource-sim --help"
echo "3. Try a simple simulation: resource-sim run --agents 10 --resources 3"
echo "4. Try a quick study: resource-sim study --config resource_allocation_sim/configs/quick_study.yaml"
echo ""
if [ "$INSTALL_DEV" = false ]; then
    echo "🛠️  For development work:"
    echo "- Install dev dependencies: ./setup.sh --dev"
    echo "- Run tests: pytest tests/ -v"
    echo ""
fi
echo "�� Happy simulating!" 