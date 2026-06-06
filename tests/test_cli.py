from pathlib import Path

from typer.testing import CliRunner

from agentic_retrieval_matrix.cli import app

FIXTURE = Path(__file__).resolve().parents[1] / "examples" / "fixture" / "benchmark.json"
runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Factorial benchmark" in result.output


def test_cli_matrix_runs_on_fixture(tmp_path: Path):
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "matrix",
            "--benchmark",
            str(FIXTURE),
            "--output",
            str(out),
            "--retrievers",
            "grep",
            "--deliveries",
            "inline",
        ],
    )
    assert result.exit_code == 0
    assert "Accuracy" in result.output or "100.0%" in result.output
    # results written
    jsons = list(out.glob("matrix_*.json"))
    assert len(jsons) == 1


def test_cli_single_runs(tmp_path: Path):
    result = runner.invoke(
        app,
        [
            "single",
            "--benchmark",
            str(FIXTURE),
            "--retriever",
            "grep",
            "--delivery",
            "file",
        ],
    )
    assert result.exit_code == 0
    assert "grep" in result.output.lower() or "file" in result.output.lower()


def test_cli_longmem_errors_on_missing(tmp_path: Path):
    result = runner.invoke(app, ["longmem", "--data-dir", str(tmp_path / "nope")])
    assert result.exit_code != 0
    msg = (result.output or "").lower()
    assert "longmemeval subset not found" in msg or "does not exist" in msg
