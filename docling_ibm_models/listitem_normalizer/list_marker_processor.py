"""
List Item Marker Processor for Docling Documents

This module provides a rule-based model to identify list item markers and
merge marker-only TextItems with their content to create proper ListItems.
"""

import re
from typing import List, Optional, Tuple, Union

from docling_core.types.doc import (
    DocItemLabel,
    DoclingDocument,
    GroupItem,
    GroupLabel,
    ListItem,
    OrderedList,
    ProvenanceItem,
    TextItem,
    UnorderedList,
)
from docling_core.types.labels import DocItemLabel


class ListItemMarkerProcessor:
    """
    A rule-based processor for identifying and processing list item markers.

    This class can:
    1. Identify various list item markers (bullets, numbers, letters)
    2. Detect marker-only TextItems followed by content TextItems
    3. Merge them into proper ListItems
    4. Group consecutive ListItems into appropriate list containers
    """

    def __init__(self):
        """Initialize the processor with marker patterns."""
        # Bullet markers (unordered lists)
        self.bullet_patterns = [
            r"[\u2022\u2023\u25E6\u2043\u204C\u204D\u2219\u25AA\u25AB\u25CF\u25CB]",  # Various bullet symbols
            r"[-*+•·‣⁃]",  # Common ASCII and Unicode bullets
            r"[►▶▸‣➤➢]",  # Arrow-like bullets
            r"[✓✔✗✘]",  # Checkmark bullets
        ]

        # Numbered markers (ordered lists)
        self.numbered_patterns = [
            r"\d+\.",  # 1. 2. 3.
            r"\d+\)",  # 1) 2) 3)
            r"\(\d+\)",  # (1) (2) (3)
            r"\[\d+\]",  # [1] [2] [3]
            r"[ivxlcdm]+\.",  # i. ii. iii. (Roman numerals lowercase)
            r"[IVXLCDM]+\.",  # I. II. III. (Roman numerals uppercase)
            r"[a-z]\.",  # a. b. c.
            r"[A-Z]\.",  # A. B. C.
            r"[a-z]\)",  # a) b) c)
            r"[A-Z]\)",  # A) B) C)
        ]

        # Compile all patterns
        self.compiled_bullet_patterns = [
            re.compile(f"^{pattern}$") for pattern in self.bullet_patterns
        ]
        self.compiled_numbered_patterns = [
            re.compile(f"^{pattern}$") for pattern in self.numbered_patterns
        ]

        self.compiled_bullet_item_patterns = [
            re.compile(f"^({pattern})\s(.+)") for pattern in self.bullet_patterns
        ]
        self.compiled_numbered_item_patterns = [
            re.compile(f"^({pattern})\s(.+)") for pattern in self.numbered_patterns
        ]

    def _is_bullet_marker(self, text: str) -> bool:
        """Check if text is a bullet marker."""
        text = text.strip()
        return any(pattern.match(text) for pattern in self.compiled_bullet_patterns)

    def _is_numbered_marker(self, text: str) -> bool:
        """Check if text is a numbered marker."""
        text = text.strip()
        return any(pattern.match(text) for pattern in self.compiled_numbered_patterns)

    def _find_marker_content_pairs(self, doc: DoclingDocument):
        """
        Find pairs of marker-only TextItems and their content TextItems.

        Returns:
            List of (marker_item, content_item) tuples. content_item can be None
            if the marker item already contains content.
        """
        self.matched_items: dict[int, tuple[str, bool]] = (
            {}
        )  # index to (self_ref, is_pure_marker)
        self.other: dict[int, str] = {}  # index to self_ref

        for i, (item, level) in enumerate(doc.iterate_items(with_groups=False)):
            if not isinstance(item, TextItem):
                continue

            if self._is_bullet_marker(item.text):
                self.matched_items[i] = (item.self_ref, True)
            elif self._is_numbered_marker(item.text):
                self.matched_items[i] = (item.self_ref, True)
            else:
                for pattern in self.compiled_bullet_item_patterns:
                    mtch = pattern.match(text)
                    if mtch:
                        self.matched_items[i] = (item.self_ref, False)

                        if isinstance(item, ListItem):
                            item.marker = mtch[1]
                            item.text = mtch[2]
                        else:
                            _log.warning(
                                f"matching text for bullet_item_patterns that is not ListItem: {item.label}"
                            )

                for pattern in self.compiled_numbered_item_patterns:
                    mtch = pattern.match(text)
                    if mtch:
                        self.matched_items[i] = (item.self_ref, False)

                        if isinstance(item, ListItem):
                            item.marker = mtch[1]
                            item.text = mtch[2]
                        else:
                            _log.warning(
                                f"matching text for numbered_item_patterns that is not ListItem: {item.label}"
                            )

            if i not in pairs:
                self.other[i] = item.self_ref

    def _group_consecutive_list_items(doc):
        """
        Might need to group list-items, not sure yet how...
        """
        return

    def process_document(self, doc: DoclingDocument) -> DoclingDocument:
        """
        Process the entire document to identify and convert list markers.

        Args:
            doc: The DoclingDocument to process

        Returns:
            The processed document (modified in-place)
        """

        def create_listitem(
            marker_text: str, content_text: str, provs: list[ProvenanceItem]
        ) -> ListItem:
            # Create new ListItem
            return ListItem(marker=marker_text, text=content_text, provs=provs)

        # Find all marker-content pairs: this function will identify text-items
        # with a marker fused into the text
        self.find_marker_content_pairs(doc)

        # If you find a sole marker-item followed by a text, there are
        # good chances we need to merge them into a list-item. This
        # function is only necessary as long as the layout-model does not
        # recognize list-items properly
        for ind, (self_ref, is_marker) in self.matched_items.items():

            if is_marker:

                marker_item = doc.resolve(self_ref)

                if ind + 1 in self.other:
                    next_item = doc.resolve(self.other[ind + 1])

                    if (isinstance(next_item, TextItem)) and (
                        next_item.label in [DocItemLabel.TEXT, DocItemLabel.LIST_ITEM]
                    ):

                        marker_text: str = marker_item.text
                        content_text: str = content_item.text
                        provs = marker_item.provs
                        provs.extend(content_item.provs)

                        list_item = create_listitem(
                            marker_text=marker_text,
                            content_text=content_text,
                            provs=provs,
                        )

                        # Insert the new ListItem
                        doc.insert_item_before_sibling(
                            new_item=list_item, sibling=marker_item
                        )

                        # Delete original items
                        items_to_delete = [marker_item, next_item]
                        doc.delete_items(node_items=items_to_delete)

        # Group consecutive list items
        self._group_consecutive_list_items(doc)

        return doc
