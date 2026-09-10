"""Compute repo statistics (LOC, test count) cited in the paper, from the
codebase itself -- never hardcoded. Writes results/supplementary/repo_stats.json.

Run: poetry run python3 scripts/compute_repo_stats.py
"""
import json
import subprocess
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
OUT = PROJ / "results/supplementary/repo_stats.json"


def loc(dir_path: Path) -> int:
    files = sorted(dir_path.rglob("*.py"))
    total = 0
    for f in files:
        total += sum(1 for _ in f.open(encoding="utf-8"))
    return total


def module_breakdown(pkg_dir: Path) -> dict[str, int]:
    out = {}
    for child in sorted(pkg_dir.iterdir()):
        if child.name.startswith("__") or child.name.startswith("."):
            continue
        if child.is_dir():
            out[child.name] = loc(child)
        elif child.suffix == ".py":
            out[child.name] = sum(1 for _ in child.open(encoding="utf-8"))
    return out


def test_count() -> int:
    result = subprocess.run(
        ["poetry", "run", "pytest", "--collect-only", "-q"],
        cwd=PROJ, capture_output=True, text=True, timeout=120,
    )
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.endswith("collected") or "collected in" in line:
            # e.g. "316 tests collected in 9.66s"
            n = int(line.split()[0])
            return n
    raise RuntimeError(f"could not parse test count from pytest output:\n{result.stdout}\n{result.stderr}")


def main() -> None:
    pkg = PROJ / "churnbench"
    tests = PROJ / "tests"

    pkg_loc = loc(pkg)
    tests_loc = loc(tests)
    n_tests = test_count()

    data = {
        "computed_by": "scripts/compute_repo_stats.py",
        "package_loc_churnbench": pkg_loc,
        "package_loc_by_module": module_breakdown(pkg),
        "test_suite_loc_tests": tests_loc,
        "test_count_pytest_collected": n_tests,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=2) + "\n")

    print(f"churnbench/ package LOC: {pkg_loc}")
    print(f"tests/ LOC: {tests_loc}")
    print(f"pytest collected: {n_tests} tests")
    print(f"written: {OUT.relative_to(PROJ)}")


if __name__ == "__main__":
    main()
