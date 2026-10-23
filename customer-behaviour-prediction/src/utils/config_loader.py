
from pathlib import Path
from src.models.model_persistance import get_project_root
import yaml

#def get_project_root() -> Path:
 #   """Return the project root directory."""
   # return Path(__file__).resolve().parents[2]

def load_config():
    """Load the project's YAML configuration."""
    project_root = Path(get_project_root()).resolve()
    config_path = project_root / "config" / "config.yaml"

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("The YAML configuration must contain a mapping.")

    return config
