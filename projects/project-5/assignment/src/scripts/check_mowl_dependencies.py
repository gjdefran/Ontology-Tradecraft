"""
Check all MOWL system dependencies.
This will identify missing or misconfigured dependencies.
"""

import sys
import subprocess

print("="*70)
print("MOWL SYSTEM DEPENDENCIES CHECK")
print("="*70)

# 1. Python version
print(f"\n1. PYTHON:")
print(f"   Version: {sys.version}")
print(f"   ✓ Python version OK" if sys.version_info >= (3, 7) else "   ✗ Python too old!")

# 2. Java
print(f"\n2. JAVA JDK:")
try:
    result = subprocess.run(['java', '-version'], capture_output=True, text=True)
    java_version = result.stderr.split('\n')[0]
    print(f"   {java_version}")
    print(f"   ✓ Java installed")
except FileNotFoundError:
    print(f"   ✗ Java NOT installed!")
    print(f"   MOWL requires Java JDK 8 or higher")
    print(f"   Install: apt-get install default-jdk  (Linux)")
    print(f"           brew install openjdk  (Mac)")

# 3. PyTorch
print(f"\n3. PYTORCH:")
try:
    import torch
    print(f"   Version: {torch.__version__}")
    print(f"   ✓ PyTorch installed")
    
    # Check CUDA availability
    print(f"\n   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"     - {torch.cuda.get_device_name(i)}")
    
    # Check MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps'):
        print(f"   MPS available: {torch.backends.mps.is_available()}")
    
    # Check what device will be used
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"\n   Default device: {device}")
    
    if device == "cpu":
        print(f"   ⚠️  WARNING: Using CPU only!")
        print(f"   Training will be MUCH slower than expected.")
        print(f"   But should still take minutes, not seconds.")
        
    # Test tensor creation
    print(f"\n   Testing tensor operations...")
    x = torch.randn(100, 100).to(device)
    y = torch.randn(100, 100).to(device)
    z = torch.mm(x, y)
    print(f"   ✓ Tensor operations work on {device}")
    
except ImportError:
    print(f"   ✗ PyTorch NOT installed!")
    print(f"   Install: pip install torch")

# 4. MOWL
print(f"\n4. MOWL:")
try:
    import mowl
    print(f"   Version: {mowl.__version__}")
    print(f"   Location: {mowl.__file__}")
    print(f"   ✓ MOWL installed")
except ImportError:
    print(f"   ✗ MOWL NOT installed!")
    print(f"   Install: pip install mowl")

# 5. JPype (Python-Java bridge)
print(f"\n5. JPYPE:")
try:
    import jpype
    print(f"   Version: {jpype.__version__}")
    print(f"   ✓ JPype installed")
    
    # Try starting JVM
    print(f"\n   Testing JVM startup...")
    if not jpype.isJVMStarted():
        import mowl
        mowl.init_jvm("2g")
        print(f"   ✓ JVM started successfully")
    else:
        print(f"   ✓ JVM already running")
        
except ImportError:
    print(f"   ✗ JPype NOT installed!")
    print(f"   Install: pip install JPype1")
except Exception as e:
    print(f"   ✗ JVM startup failed: {e}")

# 6. OWL API (via JPype)
print(f"\n6. OWL API:")
try:
    import jpype
    import jpype.imports
    if jpype.isJVMStarted():
        from org.semanticweb.owlapi.apibinding import OWLManager
        manager = OWLManager.createOWLOntologyManager()
        print(f"   ✓ OWL API accessible via JPype")
    else:
        print(f"   ⚠️  JVM not started, can't test OWL API")
except Exception as e:
    print(f"   ✗ OWL API error: {e}")

# 7. NumPy
print(f"\n7. NUMPY:")
try:
    import numpy as np
    print(f"   Version: {np.__version__}")
    print(f"   ✓ NumPy installed")
except ImportError:
    print(f"   ✗ NumPy NOT installed!")

# 8. Test actual MOWL training setup
print(f"\n8. MOWL TRAINING COMPONENTS:")
try:
    import torch
    import mowl
    from mowl.models import ELEmbeddings
    
    # Check if ELEmbeddings has proper PyTorch backend
    print(f"   Testing ELEmbeddings import...")
    print(f"   ✓ ELEmbeddings available")
    
    # Check model architecture
    print(f"\n   Checking internal model components...")
    import inspect
    
    # Check if using PyTorch properly
    elembeddings_file = inspect.getfile(ELEmbeddings)
    print(f"   Model file: {elembeddings_file}")
    
except Exception as e:
    print(f"   ✗ Error loading MOWL components: {e}")

print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)

# Determine likely issues
issues = []
recommendations = []

try:
    import torch
    device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu")
    
    if device == "cpu":
        issues.append("Using CPU instead of GPU")
        recommendations.append("Training will be slower but should still work")
        recommendations.append("Expected time with CPU: ~5-10 min for dim=200, epochs=200")
        recommendations.append("Your 'seconds' timing suggests MOWL isn't actually training")
except:
    issues.append("PyTorch not properly installed")
    recommendations.append("Install: pip install torch")

# Check the critical dependency chain
print(f"\nDependency Chain Check:")

all_ok = True
try:
    import torch
    print(f"✓ PyTorch: OK")
except:
    print(f"✗ PyTorch: MISSING")
    all_ok = False

try:
    import jpype
    print(f"✓ JPype: OK")
except:
    print(f"✗ JPype: MISSING")
    all_ok = False

try:
    import mowl
    print(f"✓ MOWL: OK")
except:
    print(f"✗ MOWL: MISSING")
    all_ok = False

try:
    import numpy
    print(f"✓ NumPy: OK")
except:
    print(f"✗ NumPy: MISSING")
    all_ok = False

if all_ok:
    print(f"\n✓ All critical dependencies installed")
    print(f"\nBut your training is still broken (seconds instead of minutes).")
    print(f"\nMost likely causes:")
    print(f"1. MOWL version bug")
    print(f"2. PyTorch not being used by MOWL (silent fallback)")
    print(f"3. Batch generation broken")
    print(f"4. Negative sampling disabled")
else:
    print(f"\n✗ Missing critical dependencies!")
    print(f"Install missing packages and try again.")

print("\n" + "="*70)
print("RECOMMENDED NEXT STEPS")
print("="*70)

if all_ok:
    print("""
Since all dependencies are installed but training is still broken:

1. TRY DIFFERENT MOWL VERSION:
   pip uninstall mowl
   pip install mowl==1.0.0
   
2. CHECK MOWL GITHUB ISSUES:
   https://github.com/bio-ontology-research-group/mowl/issues
   Search for "training too fast" or "embeddings not learning"
   
3. ENABLE MOWL DEBUG OUTPUT:
   Add to your training script:
   import logging
   logging.basicConfig(level=logging.DEBUG)
   
4. OR JUST USE LLM-ONLY FILTERING:
   This is getting ridiculous. Just use Claude for filtering.
   Set: cosine_weight=0.0, llm_weight=1.0
   Move on with your project.

HONESTLY: I recommend option 4. You've spent enough time debugging.
""")
else:
    print("""
Install missing dependencies:
pip install torch jpype1 mowl numpy

Then re-run this check.
""")

print("="*70)