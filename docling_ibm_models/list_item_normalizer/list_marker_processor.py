"""
List Item Marker Processor for Docling Documents

This module provides a rule-based model to identify list item markers and
merge marker-only TextItems with their content to create proper ListItems.
"""

import logging
import re
from typing import Union

from docling_core.types.doc.document import (
    DocItemLabel,
    DoclingDocument,
    ListItem,
    ProvenanceItem,
    RefItem,
    TextItem,
)
from docling_core.types.doc.labels import DocItemLabel

_log = logging.getLogger(__name__)


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
        self._bullet_patterns = [
            r"[\u2022\u2023\u25E6\u2043\u204C\u204D\u2219\u25AA\u25AB\u25CF\u25CB]",  # Various bullet symbols
            r"[-*+•·‣⁃]",  # Common ASCII and Unicode bullets
            r"[►▶▸‣➤➢]",  # Arrow-like bullets
            r"[✓✔✗✘]",  # Checkmark bullets
        ]

        # Numbered markers (ordered lists)
        self._numbered_patterns = [
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
        self._compiled_bullet_patterns = [
            re.compile(f"^{pattern}$") for pattern in self._bullet_patterns
        ]
        self._compiled_numbered_patterns = [
            re.compile(f"^{pattern}$") for pattern in self._numbered_patterns
        ]

        self._compiled_bullet_item_patterns = [
            re.compile(f"^({pattern})" + r"\s(.+)") for pattern in self._bullet_patterns
        ]
        self._compiled_numbered_item_patterns = [
            re.compile(f"^({pattern})" + r"\s(.+)")
            for pattern in self._numbered_patterns
        ]

        self._compiled_item_patterns = (
            self._compiled_bullet_item_patterns + self._compiled_numbered_item_patterns
        )

    def _is_bullet_marker(self, text: str) -> bool:
        """Check if text is a bullet marker."""
        text = text.strip()
        return any(pattern.match(text) for pattern in self._compiled_bullet_patterns)

    def _is_numbered_marker(self, text: str) -> bool:
        """Check if text is a numbered marker."""
        text = text.strip()
        return any(pattern.match(text) for pattern in self._compiled_numbered_patterns)

    def _find_marker_content_pairs(self, doc: DoclingDocument):
        """
        Find pairs of marker-only TextItems and their content TextItems.

        Returns:
            List of (marker_item, content_item) tuples. content_item can be None
            if the marker item already contains content.
        """
        self._matched_items: dict[int, tuple[RefItem, bool]] = (
            {}
        )  # index to (self_ref, is_pure_marker)
        self._other: dict[int, RefItem] = {}  # index to self_ref

        for i, (item, level) in enumerate(doc.iterate_items(with_groups=False)):
            if not isinstance(item, TextItem):
                continue

            if self._is_bullet_marker(item.orig):
                self._matched_items[i] = (item.get_ref(), True)
            elif self._is_numbered_marker(item.orig):
                self._matched_items[i] = (item.get_ref(), True)
            else:
                for pattern in self._compiled_item_patterns:
                    mtch = pattern.match(item.orig)
                    if mtch:
                        self._matched_items[i] = (item.get_ref(), False)

            if i not in self._matched_items:
                self._other[i] = item.get_ref()

    def _group_consecutive_list_items(self, doc: DoclingDocument) -> DoclingDocument:
        """
        Might need to group list-items, not sure yet how...
        """
        return doc

    def process_list_item(self, item: ListItem) -> ListItem:
        """Process a ListItem to extract and update marker and text from bullet/numbered patterns.

        This method applies compiled regex patterns to match bullet point or numbered list
        formatting in the original text, then updates the ListItem's marker and text fields
        accordingly.

        Args:
            item (ListItem): The list item to process, containing original text that may
                           have bullet or numbered list formatting.

        Returns:
            ListItem: The same ListItem instance with updated marker and text fields
                     if a pattern match was found, otherwise unchanged.

        Note:
            The method modifies the input item in place when a pattern matches.
            If the item is not actually a ListItem type, a warning is logged.
        """
        for pattern in self._compiled_item_patterns:
            mtch = pattern.match(item.orig)
            if mtch:
                if isinstance(item, ListItem):  # update item in place
                    item.marker = mtch[1]
                    item.text = mtch[2]
                else:
                    _log.warning(
                        f"matching text for bullet_item_patterns that is not ListItem: {item.label}"
                    )
        return item

    def process_text_item(self, item: TextItem) -> Union[TextItem, ListItem]:
        """Process a TextItem to detect and convert bullet/numbered list formatting.

        This method examines TextItem instances to determine if they contain bullet point
        or numbered list formatting. If detected and appropriate, it either updates an
        existing ListItem or converts the TextItem into a new ListItem.

        Args:
            item (TextItem): The text item to process, which may contain bullet or
                             numbered list formatting in its original text.

        Returns:
            Union[TextItem, ListItem]:
                - If item is already a ListItem: returns the updated ListItem
                - If item is a TextItem with list formatting (and not a section heading
                  or footnote): returns a new ListItem with extracted marker and text
                - Otherwise: returns the original TextItem unchanged

        Note:
            Section headings and footnotes are excluded from conversion to preserve
            their semantic meaning. A warning is logged if pattern matching occurs
            on unexpected item types.
        """
        for pattern in self._compiled_item_patterns:
            mtch = pattern.match(item.orig)
            if mtch:
                if isinstance(item, ListItem):  # update item in place
                    item.marker = mtch[1]
                    item.text = mtch[2]

                    return item
                elif isinstance(item, TextItem) and (
                    item.label
                    not in [DocItemLabel.SECTION_HEADER, DocItemLabel.FOOTNOTE]
                ):
                    # Create new ListItem
                    return ListItem(
                        self_ref=item.get_ref().cref,
                        marker=mtch[1],
                        text=mtch[2],
                        orig=item.orig,
                        prov=item.prov,
                    )
                else:
                    _log.warning(
                        f"matching text for bullet_item_patterns that is not ListItem: {item.label}"
                    )
        return item

    def update_list_items_in_place(
        self, doc: DoclingDocument, allow_textitem: bool = False
    ) -> DoclingDocument:
        for item, level in doc.iterate_items():
            if isinstance(item, ListItem):
                item = self.process_list_item(item)
            elif allow_textitem and isinstance(item, TextItem):
                item = self.process_text_item(item)

        return doc

    def merge_markers_and_text_items_into_list_items(
        self, doc: DoclingDocument
    ) -> DoclingDocument:
        def create_listitem(
            marker_text: str,
            content_text: str,
            orig_text: str,
            prov: list[ProvenanceItem],
        ) -> ListItem:
            # Create new ListItem
            return ListItem(
                self_ref="#",
                marker=marker_text,
                text=content_text,
                orig=orig_text,
                prov=prov,
            )

        # Find all marker-content pairs: this function will identify text-items
        # with a marker fused into the text
        self._find_marker_content_pairs(doc)

        # If you find a sole marker-item followed by a text, there are
        # good chances we need to merge them into a list-item. This
        # function is only necessary as long as the layout-model does not
        # recognize list-items properly
        for ind, (self_ref, is_marker) in self._matched_items.items():

            if is_marker:

                marker_item = self_ref.resolve(doc=doc)

                if ind + 1 in self._other:
                    next_item = self._other[ind + 1].resolve(doc=doc)

                    if (isinstance(next_item, TextItem)) and (
                        next_item.label in [DocItemLabel.TEXT, DocItemLabel.LIST_ITEM]
                    ):

                        marker_text: str = marker_item.text
                        content_text: str = next_item.text
                        prov = marker_item.prov
                        prov.extend(next_item.prov)

                        list_item = create_listitem(
                            marker_text=marker_text,
                            content_text=content_text,
                            orig_text=f"{marker_text} {content_text}",
                            prov=prov,
                        )

                        # Insert the new ListItem
                        doc.insert_item_before_sibling(
                            new_item=list_item, sibling=marker_item
                        )

                        # Delete original items
                        items_to_delete = [marker_item, next_item]
                        doc.delete_items(node_items=items_to_delete)

        return doc

    def process_document(
        self,
        doc: DoclingDocument,
        allow_textitem: bool = False,
        merge_items: bool = False,
    ) -> DoclingDocument:
        """
        Process the entire document to identify and convert list markers.

        Args:
            doc: The DoclingDocument to process

        Returns:
            The processed document (modified in-place)
        """
        doc = self.update_list_items_in_place(doc, allow_textitem=allow_textitem)

        if merge_items:
            doc = self.merge_markers_and_text_items_into_list_items(doc)

        # Group consecutive list items
        doc = self._group_consecutive_list_items(doc)

        return doc
