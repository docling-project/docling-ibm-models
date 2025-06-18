#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
import copy
import logging
import re
from typing import Dict, List, Set, Tuple

from docling_core.types.doc.base import BoundingBox, Size
from docling_core.types.doc.document import RefItem
from docling_core.types.doc.labels import DocItemLabel
from pydantic import BaseModel
from rtree import index as rtree_index


class PageElement(BoundingBox):

    eps: float = 1.0e-3

    cid: int
    ref: RefItem = RefItem(cref="#")  # type: ignore

    text: str = ""

    page_no: int
    page_size: Size

    label: DocItemLabel

    def __str__(self):
        return f"{self.cid:6.2f}\t{str(self.label):<10}\t{self.l:6.2f}, {self.b:6.2f}, {self.r:6.2f}, {self.t:6.2f}"

    def __lt__(self, other):
        if self.page_no == other.page_no:

            if self.overlaps_horizontally(other):
                return self.b > other.b
            else:
                return self.l < other.l
        else:
            return self.page_no < other.page_no

    def follows_maintext_order(self, rhs) -> bool:
        return self.cid + 1 == rhs.cid


class ReadingOrderPredictor:
    r"""
    Rule based reading order for DoclingDocument
    """

    def __init__(self):
        self.dilated_page_element = True

        self.initialise()

    def initialise(self):
        self.h2i_map: Dict[int, int] = {}
        self.i2h_map: Dict[int, int] = {}

        self.l2r_map: Dict[int, int] = {}
        self.r2l_map: Dict[int, int] = {}

        self.up_map: Dict[int, List[int]] = {}
        self.dn_map: Dict[int, List[int]] = {}

        self.heads: List[int] = []

    def predict_reading_order(
        self, page_elements: List[PageElement]
    ) -> List[PageElement]:

        page_nos: Set[int] = set()

        for i, elem in enumerate(page_elements):
            page_nos.add(elem.page_no)

        page_to_elems: Dict[int, List[PageElement]] = {}
        page_to_headers: Dict[int, List[PageElement]] = {}
        page_to_footers: Dict[int, List[PageElement]] = {}

        for page_no in page_nos:
            page_to_elems[page_no] = []
            page_to_footers[page_no] = []
            page_to_headers[page_no] = []

        for i, elem in enumerate(page_elements):
            if elem.label == DocItemLabel.PAGE_HEADER:
                page_to_headers[elem.page_no].append(elem)
            elif elem.label == DocItemLabel.PAGE_FOOTER:
                page_to_footers[elem.page_no].append(elem)
            else:
                page_to_elems[elem.page_no].append(elem)

        # print("headers ....")
        for page_no, elems in page_to_headers.items():
            page_to_headers[page_no] = self._predict_page(elems)

        # print("elems ....")
        for page_no, elems in page_to_elems.items():
            page_to_elems[page_no] = self._predict_page(elems)

        # print("footers ....")
        for page_no, elems in page_to_footers.items():
            page_to_footers[page_no] = self._predict_page(elems)

        sorted_elements = []
        for page_no in page_nos:
            sorted_elements.extend(page_to_headers[page_no])
            sorted_elements.extend(page_to_elems[page_no])
            sorted_elements.extend(page_to_footers[page_no])

        return sorted_elements

    def predict_to_captions(
        self, sorted_elements: List[PageElement]
    ) -> Dict[int, List[int]]:

        to_captions: Dict[int, List[int]] = {}

        page_nos: Set[int] = set()
        for i, elem in enumerate(sorted_elements):
            page_nos.add(elem.page_no)

        page_to_elems: Dict[int, List[PageElement]] = {}
        for page_no in page_nos:
            page_to_elems[page_no] = []

        for i, elem in enumerate(sorted_elements):
            page_to_elems[elem.page_no].append(elem)

        for page_no, elems in page_to_elems.items():

            page_to_captions = self._find_to_captions(
                page_elements=page_to_elems[page_no]
            )
            for key, val in page_to_captions.items():
                to_captions[key] = val

        return to_captions

    def predict_to_footnotes(
        self, sorted_elements: List[PageElement]
    ) -> Dict[int, List[int]]:

        to_footnotes: Dict[int, List[int]] = {}

        page_nos: Set[int] = set()
        for i, elem in enumerate(sorted_elements):
            page_nos.add(elem.page_no)

        page_to_elems: Dict[int, List[PageElement]] = {}
        for page_no in page_nos:
            page_to_elems[page_no] = []

        for i, elem in enumerate(sorted_elements):
            page_to_elems[elem.page_no].append(elem)

        for page_no, elems in page_to_elems.items():
            page_to_footnotes = self._find_to_footnotes(
                page_elements=page_to_elems[page_no]
            )
            for key, val in page_to_footnotes.items():
                to_footnotes[key] = val

        return to_footnotes

    def predict_merges(
        self, sorted_elements: List[PageElement]
    ) -> Dict[int, List[int]]:

        merges: Dict[int, List[int]] = {}

        curr_ind = -1
        for ind, elem in enumerate(sorted_elements):

            if ind <= curr_ind:
                continue

            if elem.label in [DocItemLabel.TEXT]:

                ind_p1 = ind + 1
                while ind_p1 < len(sorted_elements) and sorted_elements[ind_p1] in [
                    DocItemLabel.PAGE_HEADER,
                    DocItemLabel.PAGE_FOOTER,
                    DocItemLabel.TABLE,
                    DocItemLabel.PICTURE,
                    DocItemLabel.CAPTION,
                    DocItemLabel.FOOTNOTE,
                ]:
                    ind_p1 += 1

                if (
                    ind_p1 < len(sorted_elements)
                    and sorted_elements[ind_p1].label == elem.label
                    and (
                        elem.page_no != sorted_elements[ind_p1].label
                        or elem.is_strictly_left_of(sorted_elements[ind_p1])
                    )
                ):

                    m1 = re.fullmatch(r".+([a-z,\-])(\s*)", elem.text)
                    m2 = re.fullmatch(r"(\s*[a-z])(.+)", sorted_elements[ind_p1].text)

                    if m1 and m2:
                        merges[elem.cid] = [sorted_elements[ind_p1].cid]
                        curr_ind = ind_p1

        return merges

    def _predict_page(self, page_elements: List[PageElement]) -> List[PageElement]:
        r"""
        Reorder the output of the
        """

        self.initialise()

        """
        for i, elem in enumerate(page_elements):
            print(f"{i:6.2f}\t{str(elem)}")
        """

        for i, elem in enumerate(page_elements):
            page_elements[i] = elem.to_bottom_left_origin(  # type: ignore
                page_height=page_elements[i].page_size.height
            )

        self._init_h2i_map(page_elements)

        self._init_l2r_map(page_elements)

        self._init_ud_maps(page_elements)

        if self.dilated_page_element:
            dilated_page_elements: List[PageElement] = copy.deepcopy(
                page_elements
            )  # deep-copy
            dilated_page_elements = self._do_horizontal_dilation(
                page_elements, dilated_page_elements
            )

            # redo with dilated provs
            self._init_ud_maps(dilated_page_elements)

        self._find_heads(page_elements)

        self._sort_ud_maps(page_elements)

        """
        print(f"heads: {self.heads}")

        print("l2r: ")
        for k,v in self.l2r_map.items():
            print(f" -> {k}: {v}")

        print("r2l: ")
        for k,v in self.r2l_map.items():
            print(f" -> {k}: {v}")

        print("up: ")
        for k,v in self.up_map.items():
            print(f" -> {k}: {v}")

        print("dn: ")
        for k,v in self.dn_map.items():
            print(f" -> {k}: {v}")            
        """

        order: List[int] = self._find_order(page_elements)
        # print(f"order: {order}")

        sorted_elements: List[PageElement] = []
        for ind in order:
            sorted_elements.append(page_elements[ind])

        """
        for i, elem in enumerate(sorted_elements):
            print(f"{i:6.2f}\t{str(elem)}")
        """

        return sorted_elements

    def _init_h2i_map(self, page_elems: List[PageElement]):
        self.h2i_map = {}
        self.i2h_map = {}

        for i, pelem in enumerate(page_elems):
            self.h2i_map[pelem.cid] = i
            self.i2h_map[i] = pelem.cid

    def _init_l2r_map(self, page_elems: List[PageElement]):
        self.l2r_map = {}
        self.r2l_map = {}

        # this currently leads to errors ... might be necessary in the future ...
        for i, pelem_i in enumerate(page_elems):
            for j, pelem_j in enumerate(page_elems):

                if (
                    False  # pelem_i.follows_maintext_order(pelem_j)
                    and pelem_i.is_strictly_left_of(pelem_j)
                    and pelem_i.overlaps_vertically_with_iou(pelem_j, 0.8)
                ):
                    self.l2r_map[i] = j
                    self.r2l_map[j] = i

    def _init_ud_maps(self, page_elems: List[PageElement]) -> None:
        """
        Initialize up/down maps for reading order prediction using R-tree spatial indexing.

        Uses R-tree for spatial queries.
        Determines linear reading sequence by finding preceding/following elements.
        """
        self.up_map = {}
        self.dn_map = {}

        for i, pelem_i in enumerate(page_elems):
            self.up_map[i] = []
            self.dn_map[i] = []

        # Build R-tree spatial index
        spatial_idx = rtree_index.Index()
        for i, pelem in enumerate(page_elems):
            spatial_idx.insert(i, (pelem.l, pelem.b, pelem.r, pelem.t))

        for j, pelem_j in enumerate(page_elems):
            if j in self.r2l_map:
                i = self.r2l_map[j]
                self.dn_map[i] = [j]
                self.up_map[j] = [i]
                continue

            # Find elements above current that might precede it in reading order
            query_bbox = (pelem_j.l - 0.1, pelem_j.t, pelem_j.r + 0.1, float("inf"))
            candidates = list(spatial_idx.intersection(query_bbox))

            for i in candidates:
                if i == j:
                    continue

                pelem_i = page_elems[i]

                # Check spatial relationship
                if not (
                    pelem_i.is_strictly_above(pelem_j)
                    and pelem_i.overlaps_horizontally(pelem_j)
                ):
                    continue

                # Check for interrupting elements
                if not self._has_sequence_interruption(
                    spatial_idx, page_elems, i, j, pelem_i, pelem_j
                ):
                    # Follow left-to-right mapping
                    while i in self.l2r_map:
                        i = self.l2r_map[i]

                    self.dn_map[i].append(j)
                    self.up_map[j].append(i)

    def _has_sequence_interruption(
        self,
        spatial_idx: rtree_index.Index,
        page_elems: List[PageElement],
        i: int,
        j: int,
        pelem_i: PageElement,
        pelem_j: PageElement,
    ) -> bool:
        """Check if elements interrupt the reading sequence between i and j."""
        # Query R-tree for elements between i and j
        x_min = min(pelem_i.l, pelem_j.l) - 1.0
        x_max = max(pelem_i.r, pelem_j.r) + 1.0
        y_min = pelem_j.t
        y_max = pelem_i.b

        candidates = list(spatial_idx.intersection((x_min, y_min, x_max, y_max)))

        for w in candidates:
            if w in (i, j):
                continue

            pelem_w = page_elems[w]

            # Check if w interrupts the i->j sequence
            if (
                (
                    pelem_i.overlaps_horizontally(pelem_w)
                    or pelem_j.overlaps_horizontally(pelem_w)
                )
                and pelem_i.is_strictly_above(pelem_w)
                and pelem_w.is_strictly_above(pelem_j)
            ):
                return True

        return False

    def _do_horizontal_dilation(self, page_elems, dilated_page_elems):

        for i, pelem_i in enumerate(dilated_page_elems):

            x0 = pelem_i.l
            y0 = pelem_i.b

            x1 = pelem_i.r
            y1 = pelem_i.t

            if i in self.up_map and len(self.up_map[i]) > 0:
                pelem_up = page_elems[self.up_map[i][0]]

                x0 = min(x0, pelem_up.l)
                x1 = max(x1, pelem_up.r)

            if i in self.dn_map and len(self.dn_map[i]) > 0:
                pelem_dn = page_elems[self.dn_map[i][0]]

                x0 = min(x0, pelem_dn.l)
                x1 = max(x1, pelem_dn.r)

            pelem_i.l = x0
            pelem_i.r = x1

            overlaps_with_rest: bool = False
            for j, pelem_j in enumerate(page_elems):

                if i == j:
                    continue

                if not overlaps_with_rest:
                    overlaps_with_rest = pelem_j.overlaps(pelem_i)

            # update
            if not overlaps_with_rest:
                dilated_page_elems[i].l = x0
                dilated_page_elems[i].b = y0
                dilated_page_elems[i].r = x1
                dilated_page_elems[i].t = y1

        return dilated_page_elems

    def _find_heads(self, page_elems: List[PageElement]):
        head_page_elems = []
        for key, vals in self.up_map.items():
            if len(vals) == 0:
                head_page_elems.append(page_elems[key])

        """
        print("before sorting the heads: ")        
        for l, elem in enumerate(head_page_elems):
            print(f"{l}\t{str(elem)}")
        """

        # this will invoke __lt__ from PageElements
        head_page_elems = sorted(head_page_elems)

        """
        print("after sorting the heads: ")
        for l, elem in enumerate(head_page_elems):
            print(f"{l}\t{str(elem)}")
        """

        self.heads = []
        for item in head_page_elems:
            self.heads.append(self.h2i_map[item.cid])

    def _sort_ud_maps(self, provs: List[PageElement]):
        for ind_i, vals in self.dn_map.items():

            child_provs: List[PageElement] = []
            for ind_j in vals:
                child_provs.append(provs[ind_j])

            # this will invoke __lt__ from PageElements
            child_provs = sorted(child_provs)

            self.dn_map[ind_i] = []
            for child in child_provs:
                self.dn_map[ind_i].append(self.h2i_map[child.cid])

    def _find_order(self, provs: List[PageElement]):
        order: List[int] = []

        visited: List[bool] = [False for _ in provs]

        for j in self.heads:

            if not visited[j]:

                order.append(j)
                visited[j] = True
                self._depth_first_search_downwards(j, order, visited)

        if len(order) != len(provs):
            logging.error("something went wrong")

        return order

    def _depth_first_search_upwards(
        self, j: int, order: List[int], visited: List[bool]
    ):
        """depth_first_search_upwards"""

        k = j

        inds = self.up_map[j]
        for ind in inds:
            if not visited[ind]:
                return self._depth_first_search_upwards(ind, order, visited)

        return k

    def _depth_first_search_downwards(
        self, j: int, order: List[int], visited: List[bool]
    ):
        """depth_first_search_downwards"""

        inds: List[int] = self.dn_map[j]

        for i in inds:
            k: int = self._depth_first_search_upwards(i, order, visited)

            if not visited[k]:
                order.append(k)
                visited[k] = True

                self._depth_first_search_downwards(k, order, visited)

    def _find_to_captions(
        self, page_elements: List[PageElement]
    ) -> Dict[int, List[int]]:

        captions: Set[int] = set()

        # caption to picture-item/table-item
        from_captions: Dict[int, Tuple[List[int], List[int]]] = {}

        # picture-item/table-item to caption
        to_captions: Dict[int, List[int]] = {}

        # init from_captions
        for ind, page_element in enumerate(page_elements):
            if page_element.label == DocItemLabel.CAPTION:
                from_captions[page_element.cid] = ([], [])

        for ind, page_element in enumerate(page_elements):
            if page_element.label == DocItemLabel.CAPTION:
                ind_m1 = ind - 1
                while ind_m1 >= 0 and page_elements[ind_m1].label in [
                    DocItemLabel.TABLE,
                    DocItemLabel.PICTURE,
                    DocItemLabel.CODE,
                ]:
                    from_captions[page_element.cid][0].append(page_elements[ind_m1].cid)
                    ind_m1 = ind_m1 - 1

                ind_p1 = ind + 1
                while ind_p1 < len(page_elements) and page_elements[ind_p1].label in [
                    DocItemLabel.TABLE,
                    DocItemLabel.PICTURE,
                    DocItemLabel.CODE,
                ]:
                    from_captions[page_element.cid][1].append(page_elements[ind_p1].cid)
                    ind_p1 = ind_p1 + 1

        """
        for cid_i, to_item in from_captions.items():
            print("from-captions: ", cid_i, ": ", to_item[0], "; ", to_item[1])
        """

        assigned_cids = set()
        for cid_i, to_item in from_captions.items():
            if len(from_captions[cid_i][0]) == 0 and len(from_captions[cid_i][1]) > 0:
                for cid_j in from_captions[cid_i][1]:
                    # To avoid overwriting that to_captions[cid_j] when they exist
                    if to_captions.get(cid_j) is None:
                        to_captions[cid_j] = [cid_i]
                    elif cid_i not in to_captions[cid_j]:
                        to_captions[cid_j].append(cid_i)
                    # to_captions[cid_j] = [cid_i]

                    assigned_cids.add(cid_j)

            if len(from_captions[cid_i][0]) > 0 and len(from_captions[cid_i][1]) == 0:
                for cid_j in from_captions[cid_i][0]:
                    # To avoid overwriting that to_captions[cid_j] when they exist
                    if to_captions.get(cid_j) is None:
                        to_captions[cid_j] = [cid_i]
                    elif cid_i not in to_captions[cid_j]:
                        to_captions[cid_j].append(cid_i)
                    # to_captions[cid_j] = [cid_i]
                    assigned_cids.add(cid_j)

        for cid_i, to_item in from_captions.items():
            # To avoid changing the size of from_captions[cid_i][0] while iterating...
            preceding_to_remove = set()
            following_to_remove = set()

            for cid_j in from_captions[cid_i][0]:
                if cid_j in assigned_cids:
                    preceding_to_remove.add(cid_j)
                    # from_captions[cid_i][0].remove(cid_j)

            for cid_j in from_captions[cid_i][1]:
                if cid_j in assigned_cids:
                    following_to_remove.add(cid_j)
                    # from_captions[cid_i][1].remove(cid_j)

            for num in preceding_to_remove:
                from_captions[cid_i][0].remove(num)
            for num in following_to_remove:
                from_captions[cid_i][1].remove(num)

        for cid_i, to_item in from_captions.items():
            if len(from_captions[cid_i][0]) == 0 and len(from_captions[cid_i][1]) > 0:
                for cid_j in from_captions[cid_i][1]:
                    to_captions[cid_j] = [cid_i]
                    assigned_cids.add(cid_j)

            if len(from_captions[cid_i][0]) > 0 and len(from_captions[cid_i][1]) == 0:
                for cid_j in from_captions[cid_i][0]:
                    to_captions[cid_j] = [cid_i]
                    assigned_cids.add(cid_j)

        """
        for cid_i, to_item in to_captions.items():
            print("to-captions: ", cid_i, ": ", to_item)
        """

        def _remove_overlapping_indexes(mapping):
            used = set()
            result = {}
            for key, values in sorted(mapping.items()):
                valid = [
                    v
                    for v in sorted(values, key=lambda v: abs(v - key))
                    if v not in used
                ]
                if valid:
                    result[key] = [valid[0]]
                    used.add(valid[0])
            return result

        to_captions = _remove_overlapping_indexes(to_captions)
        return to_captions

    def _find_to_footnotes(
        self, page_elements: List[PageElement]
    ) -> Dict[int, List[int]]:

        to_footnotes: Dict[int, List[int]] = {}

        # Try find captions that precede the table and footnotes that come after the table
        for ind, page_element in enumerate(page_elements):

            if page_element.label in [DocItemLabel.TABLE, DocItemLabel.PICTURE]:

                ind_p1 = ind + 1
                while (
                    ind_p1 < len(page_elements)
                    and page_elements[ind_p1].label == DocItemLabel.FOOTNOTE
                ):
                    if page_element.cid in to_footnotes:
                        to_footnotes[page_element.cid].append(page_elements[ind_p1].cid)
                    else:
                        to_footnotes[page_element.cid] = [page_elements[ind_p1].cid]

                    ind_p1 += 1

        return to_footnotes
