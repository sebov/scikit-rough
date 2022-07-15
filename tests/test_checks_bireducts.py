import numpy as np
import pytest

from skrough.checks import check_if_bireduct


def _process_idx_str(idx_str):
    elements = filter(None, idx_str.split(","))
    elements = [[int(ee) for ee in e.split("..")] for e in elements]
    elements = [np.arange(eee[0] - 1, eee[-1]) if eee else [] for eee in elements]
    if elements:
        result = np.concatenate(elements)
    else:
        result = np.empty(0, dtype=np.int64)
    return result


# examples taken from the paper, thus preserved 1-based enumeration of objects
# and attrs, with use of compressed notation, e.g., 9..13
# helper function _process_idx_str translates to uncompressed 0-based


@pytest.mark.parametrize(
    "attrs, objs_multi",
    [
        (
            "",
            [
                "1,2,6,8,14",
                "3..5,7,9..13",
            ],
        ),
        (
            "3",
            [
                "1,2,5,7..11,13,14",
                "3,4,6,12",
            ],
        ),
        (
            "3,4",
            [
                "1,2,5,6,8..10,13,14",
                "1,5,6,8..10,12,13",
                "1,5,7..13",
                "2..5,7,9..11,13,14",
                "3..6,9,10,12,13",
            ],
        ),
        (
            "1",
            [
                "1..5,7,8,10,12,13",
                "1..3,6..8,12..14",
                "3,6,7,9,11..14",
            ],
        ),
        (
            "1,3",
            [
                "1..5,7..13",
                "1..4,6..9,11..13",
                "1..3,5,7..14",
                "1..3,6..9,11..14",
            ],
        ),
        (
            "1,3,4",
            [
                "1..14",
            ],
        ),
        (
            "1,2",
            [
                "1..5,7..10,12,13",
                "1..5,7,9..13",
                "1..4,6..10,12,13",
                "1..4,6,7,9..13",
                "1..3,5,7..9,12..14",
                "1..3,5,7,9,11..14",
                "1..3,6..9,12..14",
                "1..3,6,7,9,11..14",
            ],
        ),
        (
            "1,2,3",
            [
                "1..4,6..13",
                "1..3,6..14",
            ],
        ),
        (
            "1,2,4",
            [
                "1..14",
            ],
        ),
        (
            "1,4",
            [
                "1..8,10,12..14",
                "1,3..8,10..14",
                "2..7,9,10,12..14",
                "3..7,9..14",
            ],
        ),
        (
            "2",
            [
                "1,2,4,5,7,9..12",
                "1,2,4,6,10..12",
                "1,2,5,7..9,14",
                "3,4,6,10..13",
                "3,5,7..9,13,14",
                "3,6,8,13,14",
            ],
        ),
        (
            "2,3",
            [
                "1,2,4,5,7,9..13",
                "1,2,4,6,10..13",
                "1,2,6,8,10,11,13,14",
                "3,5,7..11,13,14",
                "3,6,8,10,11,13,14",
            ],
        ),
        (
            "2,3,4",
            [
                "1,2,4..6,9..13",
                "1,2,4..6,9..11,13,14",
                "1,2,4,5,7,9..11,13,14",
                "1,2,5,6,8..13",
                "1,2,5,6,8..11,13,14",
                "1,2,5,7..13",
                "2..6,9..11,13,14",
                "2,3,5,6,8..13",
                "2,3,5,6,8..11,13,14",
                "2,3,5,7..13",
                "2,3,5,7..11,13,14",
            ],
        ),
        (
            "2,4",
            [
                "1,2,4..6,9..12",
                "1,2,4..6,9,10,14",
                "1,2,4,5,7,9,10,14",
                "1,2,5,6,8,9,11,12",
                "1,2,5,6,8,9,14",
                "1,2,5,7..9,11,12",
                "2..6,9..13",
                "2..5,7,9..13",
                "2..5,7,9,10,13,14",
                "2,3,5,6,8,9,11..13",
                "2,3,5,6,8,9,13,14",
                "2,3,5,7..9,11..13",
                "2,3,5,7..9,13,14",
            ],
        ),
        (
            "4",
            [
                "1,7,8,11,12",
                "2..6,9,10,13,14",
            ],
        ),
    ],
)
def test_checks_if_bireduct_positive(attrs, objs_multi, golf_dataset_prep):
    x, x_counts, y, y_count = golf_dataset_prep
    _attrs = _process_idx_str(attrs)
    for objs in objs_multi:
        _objs = _process_idx_str(objs)
        assert check_if_bireduct(x, x_counts, y, y_count, _objs, _attrs)


@pytest.mark.parametrize(
    "attrs, objs_multi",
    [
        (
            "",
            [
                "",
                "1..14",
                "1,2,6",
            ],
        ),
        (
            "3",
            [
                "",
                "1..14",
                "7..11,13,14",
            ],
        ),
        (
            "3,4",
            [
                "",
                "1..14",
                "1,5,7..10",
            ],
        ),
        (
            "1",
            [
                "",
                "1..14",
                "1..5,13",
            ],
        ),
        (
            "1,3",
            [
                "",
                "1..14",
                "4,5,7..13",
            ],
        ),
        (
            "1,3,4",
            [
                "",
                "1..10",
            ],
        ),
        (
            "1,2",
            [
                "",
                "1..14",
                "1..10,13",
            ],
        ),
        (
            "1,2,3",
            [
                "",
                "1..14",
                "6..13",
            ],
        ),
        (
            "1,2,4",
            [
                "",
                "1..10",
            ],
        ),
        (
            "1,4",
            [
                "",
                "1..14",
                "2..7,12..14",
            ],
        ),
        (
            "2",
            [
                "",
                "1..14",
                "1,9..12",
            ],
        ),
        (
            "2,3",
            [
                "",
                "1..14",
                "3,6,8,14",
            ],
        ),
        (
            "2,3,4",
            [
                "",
                "1..14",
                "2,3..11,13,14",
            ],
        ),
        (
            "2,4",
            [
                "",
                "1..14",
                "1,9..12",
            ],
        ),
        (
            "4",
            [
                "",
                "1..14",
                "1,7,8",
            ],
        ),
    ],
)
def test_checks_if_not_bireduct(attrs, objs_multi, golf_dataset_prep):
    x, x_counts, y, y_count = golf_dataset_prep
    _attrs = _process_idx_str(attrs)
    for objs in objs_multi:
        _objs = _process_idx_str(objs)
        assert not check_if_bireduct(x, x_counts, y, y_count, _objs, _attrs)
