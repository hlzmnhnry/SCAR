import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pytest
import yaml

from scar.config import ConfigError, from_dict
from scar.config.schema import RunConfig, load_run_config

TEMPLATES = Path(__file__).resolve().parents[2] / "scar" / "config" / "templates"


@dataclass(frozen=True)
class Inner:
    value: float = 1.0
    names: tuple[str, ...] = ()


@dataclass(frozen=True)
class Outer:
    required: str
    inner: Inner = field(default_factory=Inner)
    when: date | None = None
    optional: int | None = None


def test_nested_conversion():
    outer = from_dict(Outer, {"required": "x", "inner": {"value": 2, "names": ["a", "b"]}, "when": "2022-10-19"})
    assert outer.inner == Inner(2.0, ("a", "b"))
    assert outer.when == date(2022, 10, 19)
    assert outer.optional is None


@pytest.mark.parametrize(("data", "message"), [
    ({"required": "x", "inner": {"valeu": 2}}, "inner.valeu: unknown option (did you mean 'value'?)"),
    ({}, "required: required option missing"),
    ({"required": "x", "optional": "three"}, "optional: "),
    ({"required": "x", "inner": {"value": True}}, "inner.value: expected a number"),
    ({"required": "x", "when": "yesterday"}, "when: expected a date"),
])
def test_errors_name_the_option(data, message):
    with pytest.raises(ConfigError, match=re.escape(message)):
        from_dict(Outer, data)


@pytest.mark.parametrize("template", sorted(TEMPLATES.glob("*.yaml")), ids=lambda p: p.stem)
def test_templates_are_valid(template, tmp_path):
    config = load_run_config(template)
    assert isinstance(config, RunConfig)
    assert config.base_directory == str(template.parent)


def test_paths_resolve_relative_to_the_config(tmp_path):
    data = yaml.safe_load((TEMPLATES / "csv.yaml").read_text())
    (tmp_path / "run.yaml").write_text(yaml.safe_dump(data))
    config = load_run_config(tmp_path / "run.yaml")
    assert config.path(config.sequence.poses.csv) == tmp_path / "ins.csv"
    assert config.path("/abs/file") == Path("/abs/file")
