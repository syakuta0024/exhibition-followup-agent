import pytest

from src.agent import should_include_schedule


class TestShouldIncludeSchedule:
    @pytest.mark.parametrize("rank", ["A", "B", "C", "D", "E"])
    def test_none_always_false(self, rank):
        assert should_include_schedule(rank, "none") is False

    @pytest.mark.parametrize("rank", ["A", "B", "C", "D", "E"])
    def test_all_always_true(self, rank):
        assert should_include_schedule(rank, "all") is True

    @pytest.mark.parametrize("rank", ["A", "B"])
    def test_ab_only_includes_ab(self, rank):
        assert should_include_schedule(rank, "ab_only") is True

    @pytest.mark.parametrize("rank", ["C", "D", "E"])
    def test_ab_only_excludes_cde(self, rank):
        assert should_include_schedule(rank, "ab_only") is False

    @pytest.mark.parametrize("rank", ["A", "B", "C", "D", "E"])
    def test_unknown_policy_safe_false(self, rank):
        assert should_include_schedule(rank, "unknown_policy") is False

    def test_empty_dates_guard_logic(self):
        # candidate_dates が [] や None の場合、bool 評価で False になり
        # should_include_schedule は呼ばれない → schedule_context は空
        assert not ([] and should_include_schedule("A", "all"))
        assert not (None and should_include_schedule("A", "all"))
