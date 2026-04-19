import datetime
import json
import urllib.request
from pathlib import Path
from typing import cast

from lfm_coder.sandbox import Sandbox, SandboxExecution

URL = "https://raw.githubusercontent.com/marimo-team/marimo/refs/heads/main/marimo/_runtime/packages/module_name_to_pypi_name.py"
# Resolve path relative to this script's location
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent
OUTPUT_FILE = REPO_ROOT / "src/lfm_coder/sandbox/module_mapping.json"


def fetch_mapping():
    print(f"Fetching mapping from {URL}...")
    with urllib.request.urlopen(URL) as response:
        content = response.read().decode("utf-8")
    # Skip to the function definition since Monty does not support `from __futures__ import annotation`
    start_idx = content.find("def")

    code = content[start_idx:] + "\n\nmodule_name_to_pypi_name()"

    res = cast(SandboxExecution, Sandbox().run(code))
    if res.failed or not res.result:
        print(f"Error running mapping script: {res.errors}")
        return

    mapping = res.result

    # Add metadata
    data = {
        "_metadata": {
            "source": "https://github.com/marimo-team/marimo/blob/main/marimo/_runtime/packages/module_name_to_pypi_name.py",
            "description": "Mapping from Python module names to PyPI package names, sourced from marimo.",
            "date_fetched": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
        "mapping": mapping,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Successfully saved mapping to {OUTPUT_FILE}")


if __name__ == "__main__":
    fetch_mapping()
