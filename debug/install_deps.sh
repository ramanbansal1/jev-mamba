#!/bin/bash
# Install dependencies for debugging tools

echo "Installing debugging tool dependencies..."

pip install thop
pip install pynvml

echo "Installation complete!"
echo ""
echo "You can now use:"
echo "  - python debug/profile_model.py  (for FLOPs and timing)"
echo "  - python debug/monitor_gpu.py      (for GPU monitoring)"
echo "  - python debug/torch_profiler.py   (for detailed profiling)"
