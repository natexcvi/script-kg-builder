import pytest

from training import KGDataset, KGDatasetBatchSampler


@pytest.mark.parametrize(
    "max_hops,expected_text_text_pairs",
    [
        (1, [("Bushpig", "Fish")]),
    ],
)
def test_kg_builder(max_hops, expected_text_text_pairs):
    dataset = KGDataset("testdata/dataset.csv", "testdata/images", max_hops=max_hops)
    assert expected_text_text_pairs == dataset.text_text_pairs
