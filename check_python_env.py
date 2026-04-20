"""Run in Thonny (F5) to see which Python is used and if NumPy / sklearn load."""
import sys

print("Executable:", sys.executable)
print("Version:   ", sys.version.split()[0])
print()

try:
    import numpy
    print("numpy      :", numpy.__version__)
except Exception as e:
    print("numpy      : FAILED —", e)
else:
    try:
        import numpy._core  # noqa: F401
        print("numpy._core: present (NumPy 2.x layout)")
    except ImportError:
        print("numpy._core: not used (NumPy 1.x — normal). If ML fails to load, re-run model_trainer.py here.")

try:
    import sklearn
    print("sklearn    :", sklearn.__version__)
except Exception as e:
    print("sklearn    :", e)
