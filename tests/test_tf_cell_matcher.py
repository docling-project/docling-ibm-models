from docling_ibm_models.tableformer.data_management.tf_cell_matcher import (
    find_intersection,
)


def test_find_intersection_overlapping():
    # Two overlapping bboxes ([x1, y1, x2, y2], y1 <= y2) share a rectangle.
    b1 = [0, 0, 10, 10]
    b2 = [5, 5, 15, 15]
    assert find_intersection(b1, b2) == [5, 5, 10, 10]


def test_find_intersection_disjoint_horizontally():
    # b2 sits entirely to the right of b1.
    assert find_intersection([0, 0, 10, 10], [20, 0, 30, 10]) is None
    # b2 sits entirely to the left of b1.
    assert find_intersection([20, 0, 30, 10], [0, 0, 10, 10]) is None


def test_find_intersection_disjoint_vertically():
    # b1 sits entirely below b2 (in bbox coordinates, b1[1] > b2[3]).
    assert find_intersection([0, 20, 10, 30], [0, 0, 10, 10]) is None
    # b2 sits entirely below b1 (b2[1] > b1[3]).  This is the case the buggy
    # ``b2[1] > b2[3]`` check missed: it never fired for a well-formed box, so
    # find_intersection returned a degenerate, y-inverted rectangle instead of
    # None.
    assert find_intersection([0, 0, 10, 10], [0, 20, 10, 30]) is None
