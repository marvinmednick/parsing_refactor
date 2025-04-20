import yaml
import os
from typing import Dict, Any, List
from .exceptions import ConfigError
from .debug import CategoryDebugConfig  # Import the debug config dataclass


def load_config(config_path: str) -> Dict[str, Any]:
    """Loads the main parsing configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Main configuration file not found: {config_path}")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        if not isinstance(config_data, dict):
            raise ConfigError(
                f"Configuration file {config_path} must contain a YAML dictionary (object)."
            )
        # TODO: Add more specific validation of config structure if needed
        return config_data
    except yaml.YAMLError as e:
        raise ConfigError(f"Error parsing YAML configuration file {config_path}: {e}")
    except Exception as e:
        raise ConfigError(f"Unexpected error loading configuration {config_path}: {e}")


def load_debug_config(config_path: str) -> Dict[str, CategoryDebugConfig]:
    """Loads the debug configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Debug configuration file not found: {config_path}")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)

        if not isinstance(raw_config, dict):
            raise ConfigError(
                f"Debug configuration file {config_path} must contain a YAML dictionary (object)."
            )

        debug_categories: Dict[str, CategoryDebugConfig] = {}
        for key, settings in raw_config.items():
            if not isinstance(settings, dict):
                print(
                    f"Warning: Ignoring invalid debug category '{key}'. Settings must be a dictionary."
                )
                continue
            # Use default None if keys are missing
            debug_categories[key] = CategoryDebugConfig(
                name=key,
                start_line=settings.get("start_line"),
                stop_line=settings.get("stop_line"),
                start_pattern=settings.get("start_pattern"),
                stop_pattern=settings.get("stop_pattern"),
                initially_active=settings.get(
                    "initially_active", False
                ),  # Default to inactive
            )
        return debug_categories
    except yaml.YAMLError as e:
        raise ConfigError(
            f"Error parsing YAML debug configuration file {config_path}: {e}"
        )
    except Exception as e:
        raise ConfigError(
            f"Unexpected error loading debug configuration {config_path}: {e}"
        )
