"""Tests for observation pruning: keeps observations bounded and archives evicted ones."""

import json
from pathlib import Path

import pytest

from src import config
from src.memory import prune_observations, read_archived_observations


@pytest.fixture()
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a temp data directory and patch config paths to use it."""
    monkeypatch.setattr("src.config.DATA_DIR", tmp_path)
    monkeypatch.setattr("src.config.TASKS_FILE", tmp_path / "tasks.json")
    monkeypatch.setattr("src.config.USER_CONTEXT_FILE", tmp_path / "user_context.json")
    monkeypatch.setattr(
        "src.config.OBSERVATIONS_ARCHIVE_FILE", tmp_path / "observations_archive.json"
    )
    return tmp_path


def _make_observations(count: int) -> list[dict[str, str]]:
    """Generate a list of observations with distinct dates for ordering."""
    return [
        {
            "date": f"2026-01-{i + 1:02d}",
            "observation": f"Observation #{i + 1}",
            "source": "conversation",
        }
        for i in range(count)
    ]


class TestPruneObservations:
    """Tests for prune_observations()."""

    def test_no_pruning_when_under_limit(self, data_dir: Path) -> None:
        """5 observations should all remain when max is 30."""
        observations = _make_observations(5)
        context: dict[str, object] = {
            "preferences": {},
            "priorities": {},
            "observations": observations,
            "updated_at": "2026-02-28T00:00:00+00:00",
        }
        result = prune_observations(context)
        assert len(result["observations"]) == 5  # type: ignore[arg-type]

    def test_prunes_to_max_when_over(self, data_dir: Path) -> None:
        """35 observations should be pruned to 30."""
        observations = _make_observations(35)
        context: dict[str, object] = {
            "preferences": {},
            "priorities": {},
            "observations": observations,
            "updated_at": "2026-02-28T00:00:00+00:00",
        }
        result = prune_observations(context)
        assert len(result["observations"]) == 30  # type: ignore[arg-type]

    def test_keeps_most_recent(self, data_dir: Path) -> None:
        """35 observations with distinct dates. The 30 most recent should be kept."""
        observations = _make_observations(35)
        result = prune_observations(
            {
                "preferences": {},
                "priorities": {},
                "observations": observations,
                "updated_at": None,
            }
        )
        kept = result["observations"]
        assert isinstance(kept, list)
        # Observations are ordered oldest→newest in the list.
        # Pruning removes the oldest (first) 5 entries, keeping the last 30.
        # _make_observations generates "2026-01-01" through "2026-01-35"
        assert kept[0]["date"] == "2026-01-06"  # 6th observation (index 5)
        assert kept[-1]["date"] == "2026-01-35"  # 35th observation (index 34)

    def test_archives_evicted_observations(self, data_dir: Path) -> None:
        """35 observations: the 5 oldest should be in the archive file."""
        archive_file = data_dir / "observations_archive.json"
        observations = _make_observations(35)
        prune_observations(
            {
                "preferences": {},
                "priorities": {},
                "observations": observations,
                "updated_at": None,
            },
            archive_file=archive_file,
        )
        assert archive_file.exists()
        archived = json.loads(archive_file.read_text(encoding="utf-8"))
        assert len(archived) == 5
        # Evicted are the oldest 5 (dates 01–05)
        assert archived[0]["date"] == "2026-01-01"
        assert archived[4]["date"] == "2026-01-05"

    def test_archive_appends_not_overwrites(self, data_dir: Path) -> None:
        """Two rounds of pruning should accumulate both batches in the archive."""
        archive_file = data_dir / "observations_archive.json"

        # Round 1: 35 observations → prune 5
        obs1 = _make_observations(35)
        prune_observations(
            {"observations": obs1, "updated_at": None},
            archive_file=archive_file,
        )
        archived1 = json.loads(archive_file.read_text(encoding="utf-8"))
        assert len(archived1) == 5

        # Round 2: create 35 new observations (different content)
        obs2 = [
            {
                "date": f"2026-03-{i + 1:02d}",
                "observation": f"March observation #{i + 1}",
                "source": "conversation",
            }
            for i in range(35)
        ]
        prune_observations(
            {"observations": obs2, "updated_at": None},
            archive_file=archive_file,
        )
        archived2 = json.loads(archive_file.read_text(encoding="utf-8"))
        # 5 from round 1 + 5 from round 2 = 10 total
        assert len(archived2) == 10

    def test_handles_empty_observations(self, data_dir: Path) -> None:
        """Context with empty observations list should pass through unchanged."""
        context: dict[str, object] = {
            "preferences": {},
            "priorities": {},
            "observations": [],
            "updated_at": None,
        }
        result = prune_observations(context)
        assert result["observations"] == []

    def test_handles_missing_observations_key(self, data_dir: Path) -> None:
        """Context with no observations key at all should pass through safely."""
        context: dict[str, object] = {
            "preferences": {},
            "priorities": {},
            "updated_at": None,
        }
        result = prune_observations(context)
        # Should return context as-is, no crash
        assert "observations" not in result or result["observations"] == []


class TestReadArchivedObservations:
    """Tests for read_archived_observations()."""

    def test_returns_empty_when_no_file(self, data_dir: Path) -> None:
        """No archive file exists. Should return empty list."""
        missing_file = data_dir / "nonexistent_archive.json"
        result = read_archived_observations(archive_file=missing_file)
        assert result == []

    def test_reads_existing_archive(self, data_dir: Path) -> None:
        """Reads back archived observations from file."""
        archive_file = data_dir / "observations_archive.json"
        archived = [
            {"date": "2026-01-01", "observation": "Old obs", "source": "conversation"}
        ]
        archive_file.write_text(json.dumps(archived), encoding="utf-8")
        result = read_archived_observations(archive_file=archive_file)
        assert len(result) == 1
        assert result[0]["observation"] == "Old obs"
