"""
Configuration management for vLLM-omni.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file or use defaults.
    
    Args:
        config_path: Path to configuration file. If None, loads default config.
        
    Returns:
        Configuration dictionary.
    """
    if config_path is None:
        config_path = os.path.join(
            Path(__file__).parent.parent.parent,
            "configs",
            "default.yaml"
        )
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary to save.
        config_path: Path where to save the configuration.
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration.
        override_config: Configuration to override with.
        
    Returns:
        Merged configuration dictionary.
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged
