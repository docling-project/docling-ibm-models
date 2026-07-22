import copy

import logging

import sys

import pytest

from typing import List, Dict
import random

from docling_ibm_models.reading_order.reading_order_rb import PageElement, ReadingOrderPredictor

from docling_core.types.doc.base import Size
from docling_core.types.doc.document import DoclingDocument, DocItem, TextItem, ContentLayer
from docling_core.types.doc.labels import DocItemLabel

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def rank_array(arr):
    """Compute ranks, resolving ties by averaging."""
    sorted_indices = sorted(range(len(arr)), key=lambda i: arr[i])  # Sort indices
    ranks = [0] * len(arr)  # Initialize ranks
    
    i = 0
    while i < len(arr):
        start = i
        while i + 1 < len(arr) and arr[sorted_indices[i]] == arr[sorted_indices[i + 1]]:
            i += 1  # Handle ties
        avg_rank = sum(range(start + 1, i + 2)) / (i - start + 1)  # Average rank for ties
        for j in range(start, i + 1):
            ranks[sorted_indices[j]] = avg_rank
        i += 1
    return ranks

def spearman_rank_correlation(arr1, arr2):
    assert len(arr1) == len(arr2), "Arrays must have the same length"
    
    # Compute ranks
    rank1 = rank_array(arr1)
    rank2 = rank_array(arr2)

    # Compute rank differences and apply formula
    d = [rank1[i] - rank2[i] for i in range(len(arr1))]
    d_squared_sum = sum(d_i ** 2 for d_i in d)

    n = len(arr1)
    if n > 1:
        rho = 1 - (6 * d_squared_sum) / (n * (n**2 - 1))
    else:
        rho = 0
    return rho
            
def test_readingorder():
    if sys.version_info >= (3, 14):
        pytest.skip("Pyarrow is not yet available for Python 3.14, hence we cannot load the dataset.")
    
    from datasets import load_dataset


    ro_scores, caption_scores, footnote_scores = [], [], []
    
    # Init the reading-order model
    romodel = ReadingOrderPredictor()

    ds = load_dataset("ds4sd/docling-dpbench")
    for row in ds["test"]:
        true_doc = DoclingDocument.model_validate_json(row["GroundTruthDocument"])
        
        true_elements: List[PageElement] = []
        pred_elements: List[PageElement] = []

        to_ref: Dict[int, str] = {}
        from_ref: Dict[str, int] = {}
        
        for item, level in true_doc.iterate_items(included_content_layers={ContentLayer.BODY, ContentLayer.FURNITURE}):
            if isinstance(item, DocItem):
                for prov in item.prov:

                    page_height = true_doc.pages[prov.page_no].size.height
                    bbox = prov.bbox.to_bottom_left_origin(page_height=page_height)

                    text = ""
                    if isinstance(item, TextItem):
                        text = item.text
                    
                    true_elements.append(
                        PageElement(
                            cid=len(true_elements),
                            ref=item.get_ref(),
                            text = text,
                            page_no=prov.page_no,
                            page_size = true_doc.pages[prov.page_no].size,
                            label=item.label,
                            l = bbox.l,
                            r = bbox.r,
                            b = bbox.b,
                            t = bbox.t,
                            coord_origin = bbox.coord_origin
                        )
                    )

                    to_ref[true_elements[-1].cid] = item.get_ref().cref 
                    from_ref[item.get_ref().cref] = true_elements[-1].cid 
                    
        rand_elements = copy.deepcopy(true_elements)
        random.shuffle(rand_elements)

        """
        print(f"reading {os.path.basename(filename)}")        
        for true_elem, rand_elem in zip(true_elements, rand_elements):
            print("true: ", str(true_elem), ", rand: ", str(rand_elem))
        """
        
        pred_elements = romodel.predict_reading_order(page_elements=rand_elements)
        #pred_elements = romodel.predict_page(page_elements=rand_elements)    

        assert len(pred_elements)==len(true_elements), f"{len(pred_elements)}!={len(true_elements)}"

        true_cids, pred_cids = [], []
        for true_elem, pred_elem, rand_elem in zip(true_elements,
                                                   pred_elements,
                                                   rand_elements):
            true_cids.append(true_elem.cid)
            pred_cids.append(pred_elem.cid)

        score = spearman_rank_correlation(true_cids, pred_cids)
        ro_scores.append(score)
        
        filename = row["document_id"]

        if score == 0:
            continue
        # Identify special cases ...
        if filename in ["doc_906d54a21ef3c7bfac03f4bb613b0c79ef32fdf81b362450c79e98a96f88708a_page_000001.png",  # 0.720588
                        "doc_2cd17a32ee330a239e19c915738df0c27e8ec3635a60a7e16e2a0cf3868d4af3_page_000001.png",  # 0.64920
                        "doc_bcb3dafc35b5e7476fd1b9cd6eccf5eeef936cd5b13ad846a4943f1e7797f4e9_page_000001.png",  # 0.65
                        "doc_a0edae1fa147c7bb78ebc493743a68ba4372b5ead31f2a2b146c35119462379e_page_000001.png",  # 0.82857
                        "doc_94ba5468fcb6277721947697048846dc0d0551296be3b45f5918ab857d21dcc7_page_000001.png",  # 0.857142
                        #  "doc_cbb4a13ffd01d9f777fdb939451d6a21cea1b869ee50d79581451e3601df9ec8_page_000001.png",

                        "doc_e2b604a3fb1541b82b6af8caca05682dff0c7735e0a3a4fa7c6a68246fb60e57_page_000001.png",  # 0.657142
                        "doc_827d21de372a2c26237ee1db526460851ae71c1867761776583535f532432e32_page_000001.png",  # 0.8922077
                        "doc_b862cd0d6f06c06ee5ab7729ed4e8ce58e6964eb0f1ab98b3865b57a4808216f_page_000001.png"]:  # 0.642857
            # print(f"{os.path.basename(filename)}: {score}")
            assert score>=0.60, f"reading-order score={score}>0.60"            
        else:
            assert score>=0.90, f"reading-order score={score}>0.90 for {filename}"


        true_to_captions: Dict[int, List[int]] = {}
        true_to_footnotes: Dict[int, List[int]] = {}

        total_caption_links = 0
        total_footnote_links = 0
        
        for table in true_doc.tables:
            table_cid = from_ref[table.get_ref().cref]

            true_to_captions[table_cid] = []  
            for caption in table.captions:
                caption_cid = from_ref[caption.get_ref().cref]
                true_to_captions[table_cid].append(caption_cid) 

                total_caption_links += 1

            true_to_footnotes[table_cid] = []  
            for footnote in table.footnotes:
                footnote_cid = from_ref[footnote.get_ref().cref]
                true_to_footnotes[table_cid].append(footnote_cid) 

                total_footnote_links += 1
                
        for picture in true_doc.pictures:
            picture_cid = from_ref[picture.get_ref().cref]

            true_to_captions[picture_cid] = []  
            for caption in picture.captions:
                caption_cid = from_ref[caption.get_ref().cref]
                true_to_captions[picture_cid].append(caption_cid)                 

                total_caption_links += 1

            true_to_footnotes[picture_cid] = []  
            for footnote in picture.footnotes:
                footnote_cid = from_ref[footnote.get_ref().cref]
                true_to_footnotes[picture_cid].append(footnote_cid)                 

                total_footnote_links += 1
                
        if total_caption_links>0:
            #print(" *********** ")
            
            pred_to_captions = romodel.predict_to_captions(sorted_elements=pred_elements)

            """
            for key,val in pred_to_captions.items():
                print(f"pred {key}: {val}")
            """
            
            score, total = 0.0, 0.0
            for key,val in true_to_captions.items():
                # print(f"true {key}: {val}")            
                
                total += 1.0
                if key in pred_to_captions and pred_to_captions[key]==val:
                    score += 1.0

            # print(f"to_captions: {score/total}")
            caption_scores.append(score/total)

        if total_footnote_links>0:
            # print(" *********** ")
            
            pred_to_footnotes = romodel.predict_to_footnotes(sorted_elements=pred_elements)

            """
            for key,val in pred_to_footnotes.items():
                print(f"pred {key}: {val}")
            """
            
            score, total = 0.0, 0.0
            for key,val in true_to_footnotes.items():
                # print(f"true {key}: {val}")            
                
                total += 1.0
                if key in pred_to_footnotes and pred_to_footnotes[key]==val:
                    score += 1.0

            # print(f"to_footnotes: {score/total}")
            footnote_scores.append(score/total)

        pred_merges = romodel.predict_merges(sorted_elements=pred_elements)
        # print("merges: ", pred_merges)
                    
                    
    mean_ro_score = sum(ro_scores)/len(ro_scores)
    mean_cp_score = sum(caption_scores)/len(caption_scores)
    mean_ft_score = sum(footnote_scores)/len(footnote_scores)

    assert mean_ro_score>0.95, "mean_ro_score>0.95"
    assert mean_cp_score>0.85, "mean_cp_score>0.85"
    assert mean_ft_score>0.90, "mean_ft_score>0.90"
    
    print("\n  score(reading): ", mean_ro_score)
    print("  score(caption): ", mean_cp_score)
    print("score(footnotes): ", mean_ft_score)


# ---------------------------------------------------------------------------
# Regression tests for caption <-> graphic pairing in
# ReadingOrderPredictor._find_to_captions.
#
# Pairing is label-agnostic: pictures and tables are treated identically, and
# the graphic nearest to a caption wins, with ties broken towards the preceding
# graphic. Every caption that has any adjacent graphic gets paired (captions are
# never dropped), and each graphic/caption is used at most once.
#
# The shapes below are distilled from the ground-truth of tests/data/pdf/
# 2203.01017v2.pdf in the docling repo (its dense figure appendix, pp. 1/13/14),
# which the pre-fix code mis-paired by handing a picture's caption to a nearer
# table or by dropping the first caption of a cluster.
# ---------------------------------------------------------------------------

_DUMMY_PAGE_SIZE = Size(width=100.0, height=100.0)


def _graphic(cid: int) -> PageElement:
    return PageElement(
        cid=cid, page_no=0, page_size=_DUMMY_PAGE_SIZE,
        label=DocItemLabel.PICTURE, l=0.0, r=10.0, t=10.0, b=0.0,
    )


def _table(cid: int) -> PageElement:
    return PageElement(
        cid=cid, page_no=0, page_size=_DUMMY_PAGE_SIZE,
        label=DocItemLabel.TABLE, l=0.0, r=10.0, t=10.0, b=0.0,
    )


def _caption(cid: int) -> PageElement:
    return PageElement(
        cid=cid, page_no=0, page_size=_DUMMY_PAGE_SIZE,
        label=DocItemLabel.CAPTION, l=0.0, r=10.0, t=10.0, b=0.0,
    )


def test_single_ambiguous_caption_is_assigned():
    # [G0, C1, G2]: the only caption has graphics on both sides. It must still be
    # paired with a graphic (tie broken towards the preceding graphic G0).
    elements = [_graphic(0), _caption(1), _graphic(2)]

    result = ReadingOrderPredictor()._find_to_captions(elements)

    assert result == {0: [1]}


def test_all_ambiguous_captions_are_assigned():
    # [G0, C1, G2, G3, C4, G5]: both captions are ambiguous; each must still be
    # paired with its nearest graphic (ties broken towards the preceding one).
    elements = [
        _graphic(0), _caption(1), _graphic(2),
        _graphic(3), _caption(4), _graphic(5),
    ]

    result = ReadingOrderPredictor()._find_to_captions(elements)

    assert result == {0: [1], 3: [4]}


def test_one_sided_caption_does_not_orphan_middle_caption():
    # [C0, G1, G2, C3, G4, C5]: C0 must take only its nearest graphic (G1), so
    # the middle caption C3 keeps G2 instead of being orphaned.
    elements = [
        _caption(0), _graphic(1), _graphic(2),
        _caption(3), _graphic(4), _caption(5),
    ]

    result = ReadingOrderPredictor()._find_to_captions(elements)

    assert result == {1: [0], 2: [3], 4: [5]}


def test_caption_binds_to_nearest_graphic_not_the_earlier_table():
    # [T0, P1, C2]: page 1 of 2203.01017v2 ("Figure 1: Picture of a table") — a
    # table precedes the picture the caption belongs to. The caption must bind to
    # the nearer picture P1, not the earlier table T0. The pre-fix code handed it
    # to the table (lower cid / strict adjacency), orphaning the picture.
    elements = [_table(0), _graphic(1), _caption(2)]

    result = ReadingOrderPredictor()._find_to_captions(elements)

    assert result == {1: [2]}


def test_dense_cluster_pairs_every_picture_with_its_caption():
    # Page 13 of 2203.01017v2: picture/caption pairs bunched together with a table
    # in the run:  P0 C1 | P2 C3 | T4 | P5 C6  (Figures 8, 9, 10). Every picture
    # must keep its own caption and the table must take none. The ground truth for
    # this PDF still carries the old bug here (it drops the first caption, Fig 8);
    # the fixed behaviour pairs all three — captions are never dropped.
    elements = [
        _graphic(0), _caption(1),
        _graphic(2), _caption(3),
        _table(4),
        _graphic(5), _caption(6),
    ]

    result = ReadingOrderPredictor()._find_to_captions(elements)

    assert result == {0: [1], 2: [3], 5: [6]}


"""
def test_readingorder_multipage():

    filename = Path("<json with page-elements>")
    
    # Init the reading-order model
    romodel = ReadingOrderPredictor()

    true_elements: List[PageElement] = []
    pred_elements: List[PageElement] = []
    
    with open(filename, "r") as fr:
        data = json.load(fr)
        true_elements = [PageElement.model_validate(item) for item in data]

    pred_elements = romodel.predict_reading_order(page_elements=true_elements)
    for true_elem, pred_elem in zip(true_elements, pred_elements):
        print("true: ", str(true_elem), ", pred: ", str(pred_elem))
"""
