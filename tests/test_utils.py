"""Shared helpers (cf_stability/utils.py)."""

import copy
from pathlib import Path

import pytest

from cf_stability.utils import LATE_KEYS, REPO_ROOT, config_hash, git_revision, read_json, to_plain

HASH_A, HASH_B = "a" * 40, "b" * 40
E1_RUN = REPO_ROOT / "runs" / "e1" / "follownet_highd" / "mlp" / "driver_fold0_seed0"  # a run of M4 (queue e1)
E1_OVERRIDES = {"data": "follownet_highd", "model": "mlp", "fold": 0, "seed": 0}


def repository(root: Path, head: str) -> Path:
    git = root / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text(head + "\n", encoding="utf-8")
    return git


def test_git_revision_reads_the_files_of_the_repository(tmp_path):
    assert git_revision(tmp_path) is None  # no repository

    git = repository(tmp_path / "fresh", "ref: refs/heads/main")
    assert git_revision(tmp_path / "fresh") is None  # no commit yet
    (git / "refs" / "heads" / "main").write_text(HASH_A + "\n", encoding="utf-8")
    assert git_revision(tmp_path / "fresh") == HASH_A

    git = repository(tmp_path / "packed", "ref: refs/heads/main")
    (git / "packed-refs").write_text(
        f"# pack-refs with: peeled fully-peeled sorted\n{HASH_B} refs/heads/main-old\n{HASH_A} refs/heads/main\n",
        encoding="utf-8",
    )
    assert git_revision(tmp_path / "packed") == HASH_A

    repository(tmp_path / "detached", HASH_B)
    assert git_revision(tmp_path / "detached") == HASH_B


def test_git_revision_of_a_worktree(tmp_path):
    git = repository(tmp_path / "main", "ref: refs/heads/main")
    (git / "refs" / "heads" / "topic").write_text(HASH_B + "\n", encoding="utf-8")
    private = git / "worktrees" / "topic"
    private.mkdir(parents=True)
    (private / "HEAD").write_text("ref: refs/heads/topic\n", encoding="utf-8")
    (private / "commondir").write_text("../..\n", encoding="utf-8")
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / ".git").write_text(f"gitdir: {private.as_posix()}\n", encoding="utf-8")
    assert git_revision(tree) == HASH_B


def test_config_hash_ignores_the_number_type_and_the_key_order():
    assert config_hash({"a": 60, "b": [1, 2.5]}) == config_hash({"b": [1.0, 2.5], "a": 60.0})
    assert config_hash({"a": 60}) != config_hash({"a": 61})


def test_config_hash_leaves_out_late_keys_at_their_default():
    """docs/m7_contract.md, 3.4: a key added to the training config after the runs of M4-M6 changes no
    hash while it holds its default; any other value is hashed."""
    assert LATE_KEYS == {
        ("train", "penalty", "jacobian_weight"): 0.0, ("train", "penalty", "guard"): 0.0, ("per_event_margin",): None,
    }  # fmt: skip
    base = {"seed": 0, "train": {"lr": 0.001, "penalty": {"kind": "none", "weight": 1.0}}}
    late = copy.deepcopy(base)
    late["train"]["penalty"].update(jacobian_weight=0.0, guard=0)  # 0 and 0.0 are the same setting
    snapshot = copy.deepcopy(late)
    assert config_hash(late) == config_hash(base) and late == snapshot  # the config itself is not changed
    for key, value in (("jacobian_weight", 1.0), ("guard", 3.0), ("guard", False), ("guard", None)):
        changed = copy.deepcopy(late)
        changed["train"]["penalty"][key] = value
        assert config_hash(changed) != config_hash(base), (key, value)
    # elsewhere the same names are ordinary keys
    assert config_hash({"penalty": {"guard": 0.0}}) != config_hash({"penalty": {}})
    assert config_hash({"train": {"guard": 0.0}}) != config_hash({"train": {}})
    assert config_hash([1, 2]) == config_hash([1.0, 2.0])  # not a mapping: nothing to leave out
    # D118: the per-event margin of configs/calibrate_idm.yaml, null by default (None is not 0 or false)
    calibration = {"dataset": "ngsim_i80", "per_event": True}
    assert config_hash({**calibration, "per_event_margin": None}) == config_hash(calibration)
    for value in (0.2, 0.0, False):
        assert config_hash({**calibration, "per_event_margin": value}) != config_hash(calibration), value


@pytest.mark.skipif(not (E1_RUN / "metrics.json").is_file(), reason="needs the run of E1 (configs/queue/e1.yaml)")
def test_hash_of_an_existing_run_is_unchanged():
    """The config composed now from configs/train.yaml with the overrides of a run of M4 (it holds the
    keys of M7 at their defaults) has the hash stored in the run's metrics.json, and the queue counts the
    training of the run as complete: the restartable queues do not train the runs of M4-M6 again."""
    from hydra import compose, initialize_config_dir

    from cf_stability.eval.experiment_queue import QueueConfig, load_jobs, output_problem

    metrics = read_json(E1_RUN / "metrics.json")
    assert config_hash(metrics["config"]) == metrics["config_hash"]  # the stored config with the current code
    args = [f"{key}={value}" for key, value in E1_OVERRIDES.items()] + ["experiment=e1"]
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base="1.3", job_name="test_utils"):
        plain = to_plain(compose(config_name="train", overrides=args))
        tuned = to_plain(compose(config_name="train", overrides=[*args, "train.penalty.jacobian_weight=1.0"]))
    penalty = plain["train"]["penalty"]
    assert {key: penalty[key] for key in ("jacobian_weight", "guard")} == {"jacobian_weight": 0.0, "guard": 0.0}
    assert "jacobian_weight" not in metrics["config"]["train"]["penalty"]  # the run predates the keys
    assert config_hash(plain) == metrics["config_hash"] != config_hash(tuned)
    group = {"experiment": "e1", "overrides": {key: [value] for key, value in E1_OVERRIDES.items()}}
    (job,), _ = load_jobs(QueueConfig.from_mapping({"name": "e1", "groups": [group]}))
    assert job.run_dir == E1_RUN and job.config_hash == metrics["config_hash"]
    assert output_problem(job, "train") is None
