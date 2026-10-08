import sys
import os

# Add stylegan2_ada to sys.path for imports solving (dnnlib, legacy, etc.)
_stylegan_dir = os.path.dirname(os.path.abspath(__file__))
if _stylegan_dir not in sys.path:
    sys.path.insert(0, _stylegan_dir)