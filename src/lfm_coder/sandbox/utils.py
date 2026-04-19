import ast
import json
import sys
from importlib.resources import files

from lfm_coder.logging_utils import get_logger

logger = get_logger(__name__)

# Standard library modules for PEP 723 metadata filtering and dependency detection
STD_LIB = getattr(
    sys,
    "stdlib_module_names",
    frozenset(
        [
            "abc",
            "argparse",
            "ast",
            "asyncio",
            "base64",
            "collections",
            "contextlib",
            "copy",
            "csv",
            "datetime",
            "decimal",
            "enum",
            "functools",
            "glob",
            "hashlib",
            "html",
            "http",
            "importlib",
            "inspect",
            "io",
            "itertools",
            "json",
            "logging",
            "math",
            "multiprocessing",
            "os",
            "pathlib",
            "pickle",
            "pprint",
            "queue",
            "random",
            "re",
            "shutil",
            "signal",
            "socket",
            "sqlite3",
            "statistics",
            "string",
            "struct",
            "subprocess",
            "sys",
            "tempfile",
            "threading",
            "time",
            "traceback",
            "types",
            "typing",
            "unittest",
            "urllib",
            "uuid",
            "warnings",
            "xml",
            "zipfile",
        ]
    ),
)


def load_module_mapping() -> dict[str, str]:
    """Load the module-to-PyPI package name mapping."""
    # Use importlib.resources.files to access the mapping file, which works even when
    # this package is installed (i.e., it is more robust than pathlib.Path).
    mapping_file = files("lfm_coder.sandbox").joinpath("module_mapping.json")
    try:
        with mapping_file.open("r") as f:
            data = json.load(f)
            return data.get("mapping", {})
    except Exception as e:
        logger.warning(f"Failed to load module mapping from {mapping_file}: {e}")
    return {}


def detect_dependencies(code: str, module_mapping: dict[str, str]) -> list[str]:
    """Detect PyPI dependencies from Python code."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])

    dependencies = set()
    for imp in imports:
        if imp in STD_LIB:
            continue
        pypi_name = module_mapping.get(imp, imp)
        dependencies.add(pypi_name)

    return sorted(list(dependencies))
