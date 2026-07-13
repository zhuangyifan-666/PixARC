#!/usr/bin/env python3
"""Verify duplicated JiT/PixelGen common files have matching hashes and APIs."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


COMMON_FILES = (
    "finite_difference.py",
    "scheduler.py",
    "state.py",
    "manifest.py",
    "paired_metrics.py",
    "distribution_metrics.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _argument_shape(arguments: ast.arguments) -> dict[str, Any]:
    positional = [*arguments.posonlyargs, *arguments.args]
    default_offset = len(positional) - len(arguments.defaults)
    return {
        "positional": [value.arg for value in positional],
        "posonly_count": len(arguments.posonlyargs),
        "required_positional_count": default_offset,
        "vararg": arguments.vararg.arg if arguments.vararg else None,
        "keyword_only": [value.arg for value in arguments.kwonlyargs],
        "required_keyword_only": [
            value.arg
            for value, default in zip(arguments.kwonlyargs, arguments.kw_defaults)
            if default is None
        ],
        "kwarg": arguments.kwarg.arg if arguments.kwarg else None,
    }


def _public_interface(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions: dict[str, Any] = {}
    classes: dict[str, Any] = {}
    constants: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            functions[node.name] = _argument_shape(node.args)
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            methods: dict[str, Any] = {}
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and not child.name.startswith("_"):
                    methods[child.name] = _argument_shape(child.args)
            classes[node.name] = methods
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    constants.append(target.id)
    return {
        "functions": functions,
        "classes": classes,
        "public_constants": sorted(constants),
    }


def _atomic_exclusive_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite report: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    default_root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pixarc-root", type=Path, default=default_root)
    parser.add_argument(
        "--file",
        action="append",
        dest="files",
        default=[],
        help="common filename to compare (repeatable; defaults to the contract set)",
    )
    parser.add_argument("--output-json", type=Path)
    arguments = parser.parse_args()
    root = arguments.pixarc_root.resolve(strict=True)
    filenames = tuple(arguments.files) or COMMON_FILES
    reports: dict[str, Any] = {}
    all_equal = True
    for filename in filenames:
        if Path(filename).name != filename or not filename.endswith(".py"):
            parser.error(f"unsafe common filename: {filename!r}")
        jit_path = root / "JiT" / "baselines" / "taylorseer-style" / "taylorseer_style" / filename
        pixelgen_path = root / "PixelGen" / "baselines" / "taylorseer-style" / "taylorseer_style" / filename
        jit_path.resolve(strict=True)
        pixelgen_path.resolve(strict=True)
        jit_api = _public_interface(jit_path)
        pixelgen_api = _public_interface(pixelgen_path)
        hash_equal = _sha256(jit_path) == _sha256(pixelgen_path)
        api_equal = jit_api == pixelgen_api
        reports[filename] = {
            "jit_sha256": _sha256(jit_path),
            "pixelgen_sha256": _sha256(pixelgen_path),
            "hash_equal": hash_equal,
            "public_interface_equal": api_equal,
            "jit_public_interface": jit_api,
            "pixelgen_public_interface": pixelgen_api,
        }
        all_equal = all_equal and hash_equal and api_equal
    report = {
        "pixarc_root": str(root),
        "common_files": reports,
        "all_hashes_and_interfaces_equal": all_equal,
        "runtime_cross_imports_used": False,
    }
    if arguments.output_json:
        _atomic_exclusive_json(arguments.output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not all_equal:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
