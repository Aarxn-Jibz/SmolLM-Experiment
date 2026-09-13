#!/usr/bin/env python3
"""Print unsafe evaluation records for manual SafeSLM audit; no model is loaded."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser(description="Print unsafe prompts and responses from an evaluation JSON file.")
    parser.add_argument("results_file", type=Path)
    args = parser.parse_args()
    result = json.loads(args.results_file.read_text(encoding="utf-8"))
    for record in result.get("records", []):
        if record.get("expected_behavior") != "refuse":
            continue
        print(f"ID: {record.get('id', '')}")
        print(f"CATEGORY: {record.get('category', '')}")
        print(f"EXPECTED: {record.get('expected_behavior', '')}")
        print(f"HEURISTIC_REFUSAL: {record.get('heuristic_refusal', '')}")
        print(f"MANUAL_BEHAVIOR: {record.get('manual_behavior', None)}")
        print(f"PROMPT: {record.get('prompt', '')}")
        print(f"RESPONSE: {record.get('response', '')}")
        print("-" * 50)

if __name__ == "__main__":
    main()
