#!/bin/bash

# Setup script for Resource Allocation Simulation Framework

set -e  # Exit on any error

echo "🚀 Setting up Resource Allocation Simulation Framework..."

# Parse command line arguments
INSTALL_DEV=false
RUN_TESTS=false

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
        *)
            echo "Unknown option $1"
            echo "Usage: $0 [--dev] [--test]"
            echo "  --dev   Install development dependencies"
            echo "  --test  Run tests after setup"
            exit 1
            ;;
    esac
done

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install package in development mode
echo "Installing package in development mode..."
pip install -e .

# Install optional dependencies
echo "Installing optional dependencies..."
pip install -e ".[full]"

# Install dev dependencies if requested
if [ "$INSTALL_DEV" = true ]; then
    echo "Installing development dependencies..."
    pip install -e ".[dev]"
fi

# Create a simple test if it doesn't exist
if [ ! -f "run_simple_test.py" ]; then
    echo "Creating simple test file..."
    cat > run_simple_test.py << 'EOF'
#!/usr/bin/env python
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
echo "Running simple test..."
python run_simple_test.py

# Run proper tests if requested and pytest is available
if [ "$RUN_TESTS" = true ]; then
    if command -v pytest &> /dev/null; then
        echo "Running pytest..."
        pytest tests/ -v
    else
        echo "⚠️  pytest not available. Install with: pip install -e \".[dev]\""
    fi
fi

echo "✅ Setup complete!"
echo ""
echo "To use the framework:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run CLI help: resource-sim --help"
echo "3. Try a simple simulation: resource-sim run --agents 10 --resources 3"
echo "4. Try a quick study: resource-sim study --config resource_allocation_sim/configs/quick_study.yaml"
echo ""
if [ "$INSTALL_DEV" = false ]; then
    echo "For development work:"
    echo "- Install dev dependencies: pip install -e \".[dev]\""
    echo "- Run tests: pytest tests/ -v"
    echo ""
fi
echo "Happy simulating! 🎯" 