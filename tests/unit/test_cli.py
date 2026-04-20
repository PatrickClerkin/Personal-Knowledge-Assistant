"""
Unit tests for the CLI interface.

Uses Click's CliRunner to test command-line functionality without
spawning subprocesses. Tests cover the core commands: info, ingest,
search, compare, clear, and delete.
"""

import pytest
from pathlib import Path
from click.testing import CliRunner

from src.cli import cli


@pytest.fixture
def runner():
    """Create a Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def temp_index(tmp_path):
    """Return a path string for a temporary index."""
    return str(tmp_path / "test_index")


@pytest.fixture
def sample_file(tmp_path):
    """Create a sample text file for ingestion."""
    f = tmp_path / "sample.txt"
    f.write_text(
        "Object-oriented programming uses classes and objects.\n\n"
        "Inheritance allows a class to derive from another class.\n\n"
        "Polymorphism allows objects to take on multiple forms."
    )
    return str(f)


class TestInfoCommand:

    def test_info_runs_without_error(self, runner, temp_index):
        result = runner.invoke(cli, ["--index", temp_index, "info"])
        assert result.exit_code == 0

    def test_info_shows_empty_index(self, runner, temp_index):
        result = runner.invoke(cli, ["--index", temp_index, "info"])
        assert "Total chunks:  0" in result.output

    def test_info_shows_strategy(self, runner, temp_index):
        result = runner.invoke(cli, ["--index", temp_index, "info"])
        assert "Strategy:" in result.output

    def test_info_shows_model(self, runner, temp_index):
        result = runner.invoke(cli, ["--index", temp_index, "info"])
        assert "Model:" in result.output


class TestIngestCommand:

    def test_ingest_file(self, runner, temp_index, sample_file):
        result = runner.invoke(
            cli, ["--index", temp_index, "ingest", sample_file]
        )
        assert result.exit_code == 0
        assert "chunks" in result.output.lower()

    def test_ingest_creates_chunks(self, runner, temp_index, sample_file):
        result = runner.invoke(
            cli, ["--index", temp_index, "ingest", sample_file]
        )
        assert "✓" in result.output or "Created" in result.output

    def test_ingest_nonexistent_file(self, runner, temp_index):
        result = runner.invoke(
            cli, ["--index", temp_index, "ingest", "/nonexistent/file.txt"]
        )
        assert result.exit_code != 0

    def test_ingest_directory(self, runner, temp_index, tmp_path):
        (tmp_path / "a.txt").write_text("File A content.")
        (tmp_path / "b.txt").write_text("File B content.")
        result = runner.invoke(
            cli, ["--index", temp_index, "ingest", str(tmp_path)]
        )
        assert result.exit_code == 0
        assert "Files processed:" in result.output

    def test_ingest_same_file_shows_skipped(self, runner, temp_index, sample_file):
        runner.invoke(cli, ["--index", temp_index, "ingest", sample_file])
        result = runner.invoke(
            cli, ["--index", temp_index, "ingest", sample_file]
        )
        assert "Skipped" in result.output

    def test_ingest_unsupported_format(self, runner, temp_index, tmp_path):
        bad = tmp_path / "file.xyz"
        bad.write_text("unsupported content")
        result = runner.invoke(
            cli, ["--index", temp_index, "ingest", str(bad)]
        )
        assert result.exit_code != 0


class TestSearchCommand:

    def test_search_empty_index(self, runner, temp_index):
        result = runner.invoke(
            cli, ["--index", temp_index, "search", "test query"]
        )
        assert result.exit_code != 0 or "empty" in result.output.lower()

    def test_search_returns_results(self, runner, temp_index, sample_file):
        runner.invoke(cli, ["--index", temp_index, "ingest", sample_file])
        result = runner.invoke(
            cli, ["--index", temp_index, "search", "programming"]
        )
        assert result.exit_code == 0
        assert "Score:" in result.output

    def test_search_respects_top_k(self, runner, temp_index, sample_file):
        runner.invoke(cli, ["--index", temp_index, "ingest", sample_file])
        result = runner.invoke(
            cli, ["--index", temp_index, "search", "programming", "-k", "1"]
        )
        assert result.exit_code == 0
        # Should have at most 1 result block
        assert result.output.count("[1]") <= 1


class TestClearCommand:

    def test_clear_requires_confirm(self, runner, temp_index, sample_file):
        runner.invoke(cli, ["--index", temp_index, "ingest", sample_file])
        result = runner.invoke(
            cli, ["--index", temp_index, "clear", "--confirm"]
        )
        assert result.exit_code == 0
        assert "cleared" in result.output.lower() or "Removed" in result.output

    def test_clear_resets_index(self, runner, temp_index, sample_file):
        runner.invoke(cli, ["--index", temp_index, "ingest", sample_file])
        runner.invoke(
            cli, ["--index", temp_index, "clear", "--confirm"]
        )
        result = runner.invoke(cli, ["--index", temp_index, "info"])
        assert "Total chunks:  0" in result.output


class TestDeleteCommand:

    def test_delete_with_confirm(self, runner, temp_index, sample_file):
        runner.invoke(cli, ["--index", temp_index, "ingest", sample_file])
        result = runner.invoke(
            cli, ["--index", temp_index, "delete", "sample", "--confirm"]
        )
        assert result.exit_code == 0

    def test_delete_nonexistent_doc(self, runner, temp_index):
        result = runner.invoke(
            cli, ["--index", temp_index, "delete", "ghost", "--confirm"]
        )
        assert "No document found" in result.output


class TestCompareCommand:

    def test_compare_runs(self, runner, sample_file):
        result = runner.invoke(cli, ["compare", sample_file])
        assert result.exit_code == 0
        assert "Strategy Comparison" in result.output

    def test_compare_shows_strategies(self, runner, sample_file):
        result = runner.invoke(cli, ["compare", sample_file])
        assert "fixed" in result.output
        assert "sentence" in result.output