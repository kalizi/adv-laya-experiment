import runpy
from pathlib import Path

is_current_e03 = runpy.run_path(
    str(Path(__file__).parents[1] / "tools" / "check_note_numbers.py")
)["is_current_e03"]


def test_hashed_e03_scope_is_current():
    stem = "E03_option_injection__laya-typed-decisions__massive-it__pad0__append-prepend"
    assert is_current_e03(stem)
    assert is_current_e03(stem + "__9a640a91b57048c9")
    assert not is_current_e03("E03_option_injection__laya-typed-decisions__massive-it")
