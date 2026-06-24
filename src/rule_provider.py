from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config_utils import CONFIG_DIR

DEFAULT_RULE_CONFIG_JSON = CONFIG_DIR / "rule_provider_default.json"


@dataclass(frozen=True)
class RuleProvider:
    config: dict[str, Any]

    @classmethod
    def default(cls) -> "RuleProvider":
        return cls.from_json(DEFAULT_RULE_CONFIG_JSON)

    @classmethod
    def from_json(cls, path: Path) -> "RuleProvider":
        with path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        return cls(config=config)

    @property
    def target_column(self) -> str:
        return str(self.config["schema"]["target_column"])

    @property
    def text_label_column(self) -> str:
        return str(self.config["schema"]["text_label_column"])

    def allowed_text_labels(self, target_value: int | str) -> list[str]:
        rules = self.config["alignment_rules"]["target_to_text_sentiment"]
        key = str(target_value)
        if key not in rules:
            raise ValueError(f"No text alignment rule found for target value: {target_value}")
        return [str(label) for label in rules[key]]

    def required_target_values(self) -> list[int]:
        rules = self.config["alignment_rules"]["target_to_text_sentiment"]
        return sorted(int(value) for value in rules)

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.config, handle, indent=2)
            handle.write("\n")


def load_rule_provider(rule_config: Path | None = None) -> RuleProvider:
    if rule_config is None:
        return RuleProvider.default()
    return RuleProvider.from_json(rule_config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect or export the default top-down rule provider.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_RULE_CONFIG_JSON,
        help="Path to save the default rule provider JSON.",
    )
    args = parser.parse_args()

    provider = RuleProvider.default()
    provider.to_json(args.output_json)
    print(f"Saved default rule provider to: {args.output_json}")
    print(json.dumps(provider.config, indent=2))


if __name__ == "__main__":
    main()
