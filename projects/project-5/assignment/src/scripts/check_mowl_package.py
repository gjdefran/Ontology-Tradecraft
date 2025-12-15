#!/usr/bin/env python3
"""
Check which MOWL package is installed and if it's the right one.
"""

import subprocess
import sys

print("="*70)
print("CHECKING MOWL PACKAGE")
print("="*70)

# Check for 'mowl' package
print("\n1. Checking for 'mowl' package:")
result = subprocess.run(['pip', 'show', 'mowl'], capture_output=True, text=True)
if result.returncode == 0:
    print("   ✓ 'mowl' package found:")
    print(result.stdout)
else:
    print("   ✗ 'mowl' package NOT found")

# Check for 'mowl-borg' package
print("\n2. Checking for 'mowl-borg' package:")
result = subprocess.run(['pip', 'show', 'mowl-borg'], capture_output=True, text=True)
if result.returncode == 0:
    print("   ✓ 'mowl-borg' package found:")
    print(result.stdout)
else:
    print("   ✗ 'mowl-borg' package NOT found")

# Try to import
print("\n3. Testing Python imports:")

try:
    import mowl
    print(f"   ✓ 'import mowl' works")
    print(f"     Version: {mowl.__version__ if hasattr(mowl, '__version__') else 'unknown'}")
    print(f"     Location: {mowl.__file__}")
except ImportError as e:
    print(f"   ✗ 'import mowl' FAILED: {e}")

try:
    import mowl_borg
    print(f"   ✓ 'import mowl_borg' works")
    print(f"     Version: {mowl_borg.__version__ if hasattr(mowl_borg, '__version__') else 'unknown'}")
    print(f"     Location: {mowl_borg.__file__}")
except ImportError as e:
    print(f"   ✗ 'import mowl_borg' failed: {e}")

print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)

# Determine what's wrong
try:
    import mowl
    from mowl.datasets import PathDataset
    from mowl.models import ELEmbeddings
    print("\n✓ MOWL is properly installed and importable!")
    print("  The package issue is NOT your problem.")
    print("  Something else is causing the training issue.")
except ImportError as e:
    print(f"\n✗ MOWL import failed: {e}")
    print("\n  SOLUTION:")
    print("  1. Uninstall mowl-borg: pip uninstall mowl-borg")
    print("  2. Install correct mowl: pip install mowl")
    print("  3. Or install from source: pip install git+https://github.com/bio-ontology-research-group/mowl.git")

print("\n" + "="*70)