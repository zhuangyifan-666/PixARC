import os
import subprocess
from pathlib import Path

import pytest


BASELINE_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = BASELINE_ROOT / "scripts" / "launch_4gpu_50k.sh"


def _invoke_launcher(tmp_path: Path, extra: list[str]) -> tuple[subprocess.CompletedProcess[str], Path]:
    config = tmp_path / "config.yaml"
    manifest = tmp_path / "manifest.jsonl"
    sidecar = tmp_path / "manifest.jsonl.meta.json"
    config.write_text("placeholder: true\n", encoding="utf-8")
    manifest.write_text("{}\n", encoding="utf-8")
    sidecar.write_text("{}\n", encoding="utf-8")

    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text(
        '#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" > "$FAKE_PYTHON_LOG"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    guard_log = tmp_path / "guard-argv.txt"
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "FAKE_PYTHON_LOG": str(guard_log),
            "PATH": f"{fake_bin}{os.pathsep}{environment['PATH']}",
            "SPECA_GPU_TESTS_ALLOWED": "0",
        }
    )
    command = [
        "bash",
        str(LAUNCHER),
        "--config",
        str(config),
        "--manifest",
        str(manifest),
        "--output-root",
        str(tmp_path / "output"),
        "--gpu-ids",
        "0,1,2,3",
        *extra,
    ]
    result = subprocess.run(
        command,
        cwd=BASELINE_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    return result, guard_log


def _option(arguments: list[str], name: str) -> str:
    return arguments[arguments.index(name) + 1]


def test_default_profile_forces_exact_final_50k(tmp_path: Path) -> None:
    result, guard_log = _invoke_launcher(tmp_path, [])
    assert result.returncode == 2
    assert "Refusing GPU generation" in result.stderr
    arguments = guard_log.read_text(encoding="utf-8").splitlines()
    assert _option(arguments, "--max-records") == "50000"
    assert _option(arguments, "--expected-records") == "50000"
    assert _option(arguments, "--expected-per-class") == "50"
    assert _option(arguments, "--expected-num-classes") == "1000"
    assert _option(arguments, "--expected-world-size") == "4"


@pytest.mark.parametrize(
    ("records", "per_class", "classes"),
    [("1000", "1", "1000"), ("8000", "8", "1000")],
)
def test_explicit_nonfinal_proxy_counts_reach_guard(
    tmp_path: Path, records: str, per_class: str, classes: str
) -> None:
    result, guard_log = _invoke_launcher(
        tmp_path,
        [
            "--nonfinal-proxy",
            "--expected-records",
            records,
            "--expected-per-class",
            per_class,
            "--expected-num-classes",
            classes,
        ],
    )
    assert result.returncode == 2
    assert "Refusing GPU generation" in result.stderr
    arguments = guard_log.read_text(encoding="utf-8").splitlines()
    assert _option(arguments, "--max-records") == records
    assert _option(arguments, "--expected-records") == records
    assert _option(arguments, "--expected-per-class") == per_class
    assert _option(arguments, "--expected-num-classes") == classes
    assert _option(arguments, "--expected-world-size") == "4"


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (
            [
                "--expected-records",
                "1000",
                "--expected-per-class",
                "1",
                "--expected-num-classes",
                "1000",
            ],
            "without --nonfinal-proxy",
        ),
        (["--nonfinal-proxy", "--expected-records", "1000"], "all three"),
        (
            [
                "--nonfinal-proxy",
                "--expected-records",
                "999999999999999999999999",
                "--expected-per-class",
                "1",
                "--expected-num-classes",
                "1000",
            ],
            "positive decimal integers no greater than 50000",
        ),
        (
            [
                "--nonfinal-proxy",
                "--expected-records",
                "1000",
                "--expected-per-class",
                "2",
                "--expected-num-classes",
                "1000",
            ],
            "must equal",
        ),
        (
            [
                "--nonfinal-proxy",
                "--expected-records",
                "50000",
                "--expected-per-class",
                "50",
                "--expected-num-classes",
                "1000",
            ],
            "fewer than 50000",
        ),
    ],
)
def test_proxy_profile_fails_closed(
    tmp_path: Path, extra: list[str], message: str
) -> None:
    result, guard_log = _invoke_launcher(tmp_path, extra)
    assert result.returncode == 2
    assert message in result.stderr
    assert not guard_log.exists()
