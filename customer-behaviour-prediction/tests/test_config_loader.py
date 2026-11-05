
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from src.utils.config_loader import load_config


def test_load_config_success(tmp_path):
    """Test loading a valid YAML configuration."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    expected_config = {
        "model_artifacts": {"model_dir": "model_artifacts"},
        "project": {"name": "retail_customer_prediction"},
    }

    config_path = config_dir / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(expected_config),
        encoding="utf-8",
    )

    with patch(
        "src.utils.config_loader.get_project_root",
        return_value=tmp_path,
    ):
        result = load_config()

    assert result == expected_config
    assert isinstance(result, dict)


def test_load_config_file_not_found(tmp_path):
    """Test that a missing configuration file raises FileNotFoundError."""
    with patch(
        "src.utils.config_loader.get_project_root",
        return_value=tmp_path,
    ):
        with pytest.raises(
            FileNotFoundError,
            match="Configuration file not found",
        ):
            load_config()


def test_load_config_empty_yaml(tmp_path):
    """Test that an empty YAML file raises ValueError."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text(
        "",
        encoding="utf-8",
    )

    with patch(
        "src.utils.config_loader.get_project_root",
        return_value=tmp_path,
    ):
        with pytest.raises(
            ValueError,
            match="YAML configuration must contain a mapping",
        ):
            load_config()


def test_load_config_yaml_list_instead_of_mapping(tmp_path):
    """Test that a YAML list is rejected because a mapping is required."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text(
        "- classification\n- regression\n",
        encoding="utf-8",
    )

    with patch(
        "src.utils.config_loader.get_project_root",
        return_value=tmp_path,
    ):
        with pytest.raises(
            ValueError,
            match="YAML configuration must contain a mapping",
        ):
            load_config()
