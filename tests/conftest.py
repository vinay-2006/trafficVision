import os
import tempfile
from pathlib import Path

# Ensure Matplotlib uses a writable directory in all test environments
cache_dir = Path(__file__).parent.parent / ".cache" / "matplotlib"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

# Disable Ultralytics parent git traversal to avoid WinError 5 in sandboxed environments
try:
    from ultralytics.utils.git import GitRepo
    GitRepo.is_repo = property(lambda self: False)
except ImportError:
    pass
