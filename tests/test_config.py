from pathlib import Path

import pytest

from spend_engine.config import ConfigurationError, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_company_templates_load():
    load_config(ROOT / "configs" / "company_a.yaml")
    load_config(ROOT / "configs" / "company_b.yaml")


def test_c_template_is_not_runnable_until_configured():
    with pytest.raises(ConfigurationError):
        load_config(ROOT / "configs" / "company_c.template.yaml")
