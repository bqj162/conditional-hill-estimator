from pathlib import Path
import yaml
from typing import Any, Dict, Optional

def load_config(path: Optional[Path]) -> Dict[str, Any]:
    """
    Load YAML config and return as dict. If path is None, return empty dict.
    """
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg
