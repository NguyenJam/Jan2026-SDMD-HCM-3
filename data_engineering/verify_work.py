"""Run the complete offline verification and preserve its actual results."""

import io
import json
import subprocess
import sys
import unittest

from fetch_data import ROOT


def main():
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    summary_path = output / "test_summary.json"
    summary_path.write_text(
        json.dumps({"passed": False, "stage": "analysis", "tests_run": 0}, indent=2) + "\n",
        encoding="utf-8",
    )
    analysis = subprocess.run(
        [sys.executable, str(ROOT / "run_analysis.py")],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
    )
    (output / "analysis_run.txt").write_text(analysis.stdout + analysis.stderr, encoding="utf-8")
    if analysis.returncode:
        print(analysis.stdout + analysis.stderr)
        return analysis.returncode
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    log = stream.getvalue()
    (output / "test_results.txt").write_text(log, encoding="utf-8")
    summary = {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "passed": result.wasSuccessful(),
        "analysis_exit_code": analysis.returncode,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(log)
    print(json.dumps(summary, indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
