#!/usr/bin/env python3
"""
NavCore PR Review Bot — Comment Generator
==========================================
Parses CI output files (flake8, mypy, pytest, jacobian tests)
and generates a structured Markdown review comment for GitHub PRs.

Usage:
    python pr_review_comment.py \
        --lint-report reports/flake8.txt \
        --mypy-report reports/mypy.txt \
        --pytest-report reports/pytest.txt \
        --jacobian-report reports/jacobians.txt \
        --output review_comment.md
"""

import argparse
import re
import sys
from pathlib import Path
from datetime import datetime, timezone


def parse_flake8(report_path: str) -> dict:
    """Parse flake8 output into structured issues."""
    result = {"issues": [], "count": 0, "status": "✅"}

    path = Path(report_path)
    if not path.exists() or path.stat().st_size == 0:
        return result

    issues = []
    for line in path.read_text(encoding="utf-8", errors="replace").strip().splitlines():
        # Format: path:row:col: CODE message
        match = re.match(r"^(.+?):(\d+):(\d+):\s+([\w\d]+)\s+(.+)$", line.strip())
        if match:
            issues.append({
                "file": match.group(1),
                "line": int(match.group(2)),
                "col": int(match.group(3)),
                "code": match.group(4),
                "message": match.group(5),
            })

    result["issues"] = issues
    result["count"] = len(issues)
    result["status"] = "✅" if len(issues) == 0 else "⚠️"
    return result


def parse_mypy(report_path: str) -> dict:
    """Parse mypy output into structured errors."""
    result = {"errors": [], "count": 0, "status": "✅"}

    path = Path(report_path)
    if not path.exists() or path.stat().st_size == 0:
        return result

    errors = []
    content = path.read_text(encoding="utf-8", errors="replace")
    for line in content.strip().splitlines():
        # Format: file.py:line: error: message  [code]
        match = re.match(r"^(.+?):(\d+):\s+(error|warning|note):\s+(.+)$", line.strip())
        if match:
            errors.append({
                "file": match.group(1),
                "line": int(match.group(2)),
                "level": match.group(3),
                "message": match.group(4),
            })

    actual_errors = [e for e in errors if e["level"] == "error"]
    result["errors"] = errors
    result["count"] = len(actual_errors)
    result["status"] = "✅" if len(actual_errors) == 0 else "⚠️"
    return result


def parse_pytest(report_path: str) -> dict:
    """Parse pytest console output for pass/fail summary."""
    result = {
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "total": 0,
        "status": "✅",
        "failures": [],
    }

    path = Path(report_path)
    if not path.exists() or path.stat().st_size == 0:
        result["status"] = "❓"
        return result

    content = path.read_text(encoding="utf-8", errors="replace")

    # Parse the summary line: "= X passed, Y failed, Z error ="
    summary_match = re.search(
        r"=+\s*(.*?)\s*=+\s*$", content, re.MULTILINE
    )
    if summary_match:
        summary = summary_match.group(1)
        passed = re.search(r"(\d+)\s+passed", summary)
        failed = re.search(r"(\d+)\s+failed", summary)
        errors = re.search(r"(\d+)\s+error", summary)
        skipped = re.search(r"(\d+)\s+skipped", summary)

        result["passed"] = int(passed.group(1)) if passed else 0
        result["failed"] = int(failed.group(1)) if failed else 0
        result["errors"] = int(errors.group(1)) if errors else 0
        result["skipped"] = int(skipped.group(1)) if skipped else 0

    result["total"] = (
        result["passed"] + result["failed"] +
        result["errors"] + result["skipped"]
    )

    # Collect FAILED test names
    for match in re.finditer(r"FAILED\s+(.+?)(?:\s+-|$)", content):
        result["failures"].append(match.group(1).strip())

    if result["failed"] > 0 or result["errors"] > 0:
        result["status"] = "❌"
    elif result["total"] == 0:
        result["status"] = "❓"

    return result


def parse_jacobians(report_path: str) -> dict:
    """Parse Jacobian validation test output."""
    result = {"status": "✅", "details": "", "tests": []}

    path = Path(report_path)
    if not path.exists() or path.stat().st_size == 0:
        result["status"] = "❓"
        result["details"] = "Jacobian tests did not produce output."
        return result

    content = path.read_text(encoding="utf-8", errors="replace")

    # Collect individual test results
    for match in re.finditer(
        r"(tests/test_jacobians\.py::.*?)\s+(PASSED|FAILED|ERROR)", content
    ):
        result["tests"].append({
            "name": match.group(1),
            "status": match.group(2),
        })

    if "FAILED" in content:
        result["status"] = "❌"
        result["details"] = "One or more Jacobian validation tests failed."
    elif "passed" in content:
        result["status"] = "✅"
        result["details"] = "All analytical Jacobians match numerical finite-difference references."
    else:
        result["status"] = "⚠️"
        result["details"] = "Could not determine Jacobian test results."

    return result


def categorize_lint_issues(issues: list) -> dict:
    """Group lint issues by severity category."""
    categories = {
        "errors": [],      # E9xx syntax errors, F-codes
        "warnings": [],    # W-codes, E-codes
        "style": [],       # Formatting
    }

    for issue in issues:
        code = issue["code"]
        if code.startswith("F") or code.startswith("E9"):
            categories["errors"].append(issue)
        elif code.startswith("E"):
            categories["warnings"].append(issue)
        else:
            categories["style"].append(issue)

    return categories


def generate_comment(
    lint: dict,
    mypy: dict,
    pytest_result: dict,
    jacobians: dict,
) -> str:
    """Generate the full Markdown review comment."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = []
    lines.append("# 🤖 NavCore PR Review Bot")
    lines.append("")
    lines.append(f"*Automated review generated at {now}*")
    lines.append("")

    # ── Summary Table ──────────────────────────────────────
    lines.append("## Summary")
    lines.append("")
    lines.append("| Check | Status | Details |")
    lines.append("|-------|--------|---------|")

    # Lint
    lint_detail = f"{lint['count']} issue(s)" if lint['count'] > 0 else "Clean"
    lines.append(f"| 🔍 Lint (flake8) | {lint['status']} | {lint_detail} |")

    # Type check
    mypy_detail = f"{mypy['count']} error(s)" if mypy['count'] > 0 else "Clean"
    lines.append(f"| 🔠 Type Check (mypy) | {mypy['status']} | {mypy_detail} |")

    # Tests
    test_detail = (
        f"{pytest_result['passed']} passed, "
        f"{pytest_result['failed']} failed, "
        f"{pytest_result['skipped']} skipped"
    )
    lines.append(f"| 🧪 Tests (pytest) | {pytest_result['status']} | {test_detail} |")

    # Jacobians
    lines.append(f"| 📐 Jacobian Validation | {jacobians['status']} | {jacobians['details'][:80]} |")

    lines.append("")

    # ── Overall Verdict ────────────────────────────────────
    all_pass = all(
        s["status"] == "✅"
        for s in [lint, mypy, pytest_result, jacobians]
    )

    if all_pass:
        lines.append("> ✅ **All checks passed.** This PR looks good to merge.")
    elif pytest_result["status"] == "❌" or jacobians["status"] == "❌":
        lines.append("> ❌ **Tests are failing.** Please fix the issues below before merging.")
    else:
        lines.append("> ⚠️ **Some checks have warnings.** Review the details below.")

    lines.append("")

    # ── Lint Details ───────────────────────────────────────
    if lint["count"] > 0:
        lines.append("---")
        lines.append("")
        lines.append("## 🔍 Lint Issues")
        lines.append("")

        categories = categorize_lint_issues(lint["issues"])

        if categories["errors"]:
            lines.append("### ❌ Errors (must fix)")
            lines.append("")
            lines.append("| File | Line | Code | Message |")
            lines.append("|------|------|------|---------|")
            for issue in categories["errors"][:20]:
                lines.append(
                    f"| `{issue['file']}` | {issue['line']} | "
                    f"`{issue['code']}` | {issue['message']} |"
                )
            lines.append("")

        if categories["warnings"]:
            lines.append(f"### ⚠️ Warnings ({len(categories['warnings'])} total)")
            lines.append("")
            lines.append("<details>")
            lines.append("<summary>Click to expand</summary>")
            lines.append("")
            lines.append("| File | Line | Code | Message |")
            lines.append("|------|------|------|---------|")
            for issue in categories["warnings"][:30]:
                lines.append(
                    f"| `{issue['file']}` | {issue['line']} | "
                    f"`{issue['code']}` | {issue['message']} |"
                )
            if len(categories["warnings"]) > 30:
                lines.append(
                    f"| ... | ... | ... | "
                    f"*{len(categories['warnings']) - 30} more* |"
                )
            lines.append("")
            lines.append("</details>")
            lines.append("")

        if categories["style"]:
            lines.append(f"### 💅 Style ({len(categories['style'])} total)")
            lines.append("")
            lines.append("<details>")
            lines.append("<summary>Click to expand</summary>")
            lines.append("")
            lines.append("| File | Line | Code | Message |")
            lines.append("|------|------|------|---------|")
            for issue in categories["style"][:20]:
                lines.append(
                    f"| `{issue['file']}` | {issue['line']} | "
                    f"`{issue['code']}` | {issue['message']} |"
                )
            lines.append("")
            lines.append("</details>")
            lines.append("")

    # ── Type Check Details ─────────────────────────────────
    if mypy["count"] > 0:
        lines.append("---")
        lines.append("")
        lines.append("## 🔠 Type Check Issues")
        lines.append("")
        lines.append("| File | Line | Level | Message |")
        lines.append("|------|------|-------|---------|")
        for err in mypy["errors"][:20]:
            level_icon = {"error": "❌", "warning": "⚠️", "note": "ℹ️"}.get(
                err["level"], "❓"
            )
            lines.append(
                f"| `{err['file']}` | {err['line']} | "
                f"{level_icon} {err['level']} | {err['message']} |"
            )
        lines.append("")

    # ── Test Failure Details ───────────────────────────────
    if pytest_result["failed"] > 0 or pytest_result["errors"] > 0:
        lines.append("---")
        lines.append("")
        lines.append("## 🧪 Test Failures")
        lines.append("")
        for failure in pytest_result["failures"][:10]:
            lines.append(f"- ❌ `{failure}`")
        lines.append("")

    # ── Jacobian Details ───────────────────────────────────
    if jacobians["tests"]:
        lines.append("---")
        lines.append("")
        lines.append("## 📐 Jacobian Validation Details")
        lines.append("")
        lines.append("| Test | Status |")
        lines.append("|------|--------|")
        for test in jacobians["tests"]:
            icon = {"PASSED": "✅", "FAILED": "❌", "ERROR": "💥"}.get(
                test["status"], "❓"
            )
            # Shorten test name
            name = test["name"].replace("tests/test_jacobians.py::", "")
            lines.append(f"| `{name}` | {icon} {test['status']} |")
        lines.append("")

        if jacobians["status"] == "✅":
            lines.append(
                "> 📐 All analytical Jacobians match their numerical "
                "finite-difference counterparts within tolerance. "
                "The sensor update math is verified."
            )
            lines.append("")

    # ── Footer ─────────────────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append(
        "*🤖 Generated by [NavCore PR Review Bot]"
        "(https://github.com/ARYA-mgc/NavCore-Pixhawk/blob/main/"
        ".github/workflows/pr-review.yml) — "
        "runs on every PR targeting `src/`, `tests/`, or `scripts/`.*"
    )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate PR review comment from CI reports"
    )
    parser.add_argument(
        "--lint-report", required=True,
        help="Path to flake8 output file"
    )
    parser.add_argument(
        "--mypy-report", required=True,
        help="Path to mypy output file"
    )
    parser.add_argument(
        "--pytest-report", required=True,
        help="Path to pytest console output file"
    )
    parser.add_argument(
        "--jacobian-report", required=True,
        help="Path to Jacobian test output file"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output path for the review comment markdown"
    )
    args = parser.parse_args()

    # Parse all reports
    lint = parse_flake8(args.lint_report)
    mypy = parse_mypy(args.mypy_report)
    pytest_result = parse_pytest(args.pytest_report)
    jacobians = parse_jacobians(args.jacobian_report)

    # Generate comment
    comment = generate_comment(lint, mypy, pytest_result, jacobians)

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(comment, encoding="utf-8")

    print(f"Review comment generated: {output_path}")
    print(f"  Lint: {lint['status']} ({lint['count']} issues)")
    print(f"  MyPy: {mypy['status']} ({mypy['count']} errors)")
    print(f"  Tests: {pytest_result['status']} ({pytest_result['passed']} passed, {pytest_result['failed']} failed)")
    print(f"  Jacobians: {jacobians['status']}")

    # Exit with non-zero if critical failures
    if pytest_result["status"] == "❌":
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
