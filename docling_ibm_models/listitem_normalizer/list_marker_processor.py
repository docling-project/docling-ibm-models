"""
List Item Marker Processor for Docling Documents

This module provides a rule-based model to identify list item markers and
merge marker-only TextItems with their content to create proper ListItems.
"""

import re
from typing import List, Optional, Tuple, Union
from docling_core.types.doc import (
    DoclingDocument, 
    TextItem, 
    ListItem, 
    DocItemLabel,
    ProvenanceItem,
    GroupItem,
    OrderedList,
    UnorderedList,
    GroupLabel
)


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
            r'[\u2022\u2023\u25E6\u2043\u204C\u204D\u2219\u25AA\u25AB\u25CF\u25CB]',  # Various bullet symbols
            r'[-*+•·‣⁃]',  # Common ASCII and Unicode bullets
            r'[►▶▸‣➤➢]',  # Arrow-like bullets
            r'[✓✔✗✘]',  # Checkmark bullets
        ]
        
        # Numbered markers (ordered lists)
        self.numbered_patterns = [
            r'\d+\.',  # 1. 2. 3.
            r'\d+\)',  # 1) 2) 3)
            r'\(\d+\)',  # (1) (2) (3)
            r'\[\d+\]',  # [1] [2] [3]
            r'[ivxlcdm]+\.',  # i. ii. iii. (Roman numerals lowercase)
            r'[IVXLCDM]+\.',  # I. II. III. (Roman numerals uppercase)
            r'[a-z]\.',  # a. b. c.
            r'[A-Z]\.',  # A. B. C.
            r'[a-z]\)',  # a) b) c)
            r'[A-Z]\)',  # A) B) C)
        ]
        
        # Compile all patterns
        self.compiled_bullet_patterns = [re.compile(f"^{pattern}$") for pattern in self.bullet_patterns]
        self.compiled_numbered_patterns = [re.compile(f"^{pattern}$") for pattern in self.numbered_patterns]

        self.compiled_bullet_item_patterns = [re.compile(f"^({pattern})\s(.+)") for pattern in self.bullet_patterns]
        self.compiled_numbered_item_patterns = [re.compile(f"^({pattern})\s(.+)") for pattern in self.numbered_patterns]
        
    def _is_bullet_marker(self, text: str) -> bool:
        """Check if text is a bullet marker."""
        text = text.strip()
        return any(pattern.match(text) for pattern in self.compiled_bullet_patterns)
    
    def _is_numbered_marker(self, text: str) -> bool:
        """Check if text is a numbered marker."""
        text = text.strip()
        return any(pattern.match(text) for pattern in self.compiled_numbered_patterns)
    
    def _is_list_marker(self, text: str) -> Tuple[bool, bool]:
        """
        Check if text is a list marker.
        
        Returns:
            Tuple[bool, bool]: (is_marker, is_enumerated)
        """
        if self._is_numbered_marker(text):
            return True, True
        elif self._is_bullet_marker(text):
            return True, False
        return False, False
    
    def _extract_marker_and_content(self, text: str) -> Tuple[Optional[str], Optional[str], bool]:
        """
        Extract marker and content from text that might contain both.
        
        Returns:
            Tuple[Optional[str], Optional[str], bool]: (marker, content, is_enumerated)
        """
        text = text.strip()
        if not text:
            return None, None, False
        
        # Try to find marker at the beginning of text
        for pattern in self.compiled_numbered_patterns:
            match = pattern.match(text)
            if match:
                marker = match.group()
                content = text[len(marker):].strip()
                return marker, content if content else None, True
        
        for pattern in self.compiled_bullet_patterns:
            match = pattern.match(text)
            if match:
                marker = match.group()
                content = text[len(marker):].strip()
                return marker, content if content else None, False
        
        # Check if text starts with common patterns followed by space
        numbered_with_space = re.match(r'^(\d+[\.\)]|\([a-zA-Z0-9]+\)|[a-zA-Z][\.\)])\s+(.+)$', text)
        if numbered_with_space:
            return numbered_with_space.group(1), numbered_with_space.group(2), True
        
        bullet_with_space = re.match(r'^([-*+•·‣⁃►▶▸‣➤➢✓✔✗✘\u2022\u2023\u25E6\u2043\u204C\u204D\u2219\u25AA\u25AB\u25CF\u25CB])\s+(.+)$', text)
        if bullet_with_space:
            return bullet_with_space.group(1), bullet_with_space.group(2), False
        
        return None, None, False
    
    def _find_marker_content_pairs(self, doc: DoclingDocument) -> List[Tuple[TextItem, Optional[TextItem]]]:
        """
        Find pairs of marker-only TextItems and their content TextItems.
        
        Returns:
            List of (marker_item, content_item) tuples. content_item can be None
            if the marker item already contains content.
        """
        pairs: dict[int, tuple[str, bool]] = {} # index to (self_ref, is_pure_marker)
        other: dict[int, str] = {} # index to self_ref
        
        for i, (item, level) in enumerate(doc.iterate_items(with_groups=False))):
            if not isinstance(item, TextItem):
                continue

            if self._is_bullet_marker(item.text):
                pairs[i] = (item.self_ref, True)
            elif self._is_numbered_marker(item.text):
                pairs[i] = (item.self_ref, True)
            else:
                for pattern in self.compiled_bullet_item_patterns:
                    mtch = pattern.match(text)
                    if mtch:                        
                        pairs[i] = (item.self_ref, False)

                        if isinstance(item, ListItem):
                            item.marker = mtch[1]
                            item.text = mtch[2]
                        else:
                            _log.warning(f"matching text for compiled_bullet_item_patterns that is not ListItem")

                for pattern in self.compiled_numbered_item_patterns:
                    mtch = pattern.match(text)
                    if mtch:                        
                        pairs[i] = (item.self_ref, False)

                        if isinstance(item, ListItem):
                            item.marker = mtch[1]
                            item.text = mtch[2]
                        else:
                            _log.warning(f"matching text for compiled_bullet_item_patterns that is not ListItem")

            if i not in pairs:
                other[i] = item.self_ref

        return pairs
    
    def create_list_item_from_pair(self, marker_item: TextItem, content_item: Optional[TextItem]) -> ListItem:
        """Create a ListItem from a marker-content pair."""
        if content_item is None:
            # Marker and content are in the same item
            marker, content, is_enumerated = self.extract_marker_and_content(marker_item.text)
            full_text = content or ""
            orig_text = marker_item.orig
        else:
            # Separate marker and content items
            marker, _, is_enumerated = self.extract_marker_and_content(marker_item.text)
            full_text = content_item.text
            orig_text = f"{marker_item.orig} {content_item.orig}"
        
        # Create ListItem
        list_item = ListItem(
            text=full_text,
            orig=orig_text,
            self_ref="#",  # Will be updated when added to document
            parent=marker_item.parent,
            enumerated=is_enumerated,
            marker=marker or "",
            prov=marker_item.prov.copy(),
            content_layer=marker_item.content_layer,
            formatting=marker_item.formatting,
            hyperlink=marker_item.hyperlink
        )
        
        # Merge provenance from content item if exists
        if content_item and content_item.prov:
            list_item.prov.extend(content_item.prov)
        
        return list_item
    
    def group_consecutive_list_items(self, doc: DoclingDocument) -> None:
        """Group consecutive ListItems into appropriate list containers."""
        items = list(doc.iterate_items(with_groups=True))
        i = 0
        
        while i < len(items):
            item, level = items[i]
            
            if isinstance(item, ListItem):
                # Found a list item, look for consecutive ones
                consecutive_items = [item]
                list_type = GroupLabel.ORDERED_LIST if item.enumerated else GroupLabel.LIST
                j = i + 1
                
                # Collect consecutive list items of the same type at the same level
                while j < len(items):
                    next_item, next_level = items[j]
                    if (isinstance(next_item, ListItem) and 
                        next_level == level and
                        ((next_item.enumerated and list_type == GroupLabel.ORDERED_LIST) or
                         (not next_item.enumerated and list_type == GroupLabel.LIST))):
                        consecutive_items.append(next_item)
                        j += 1
                    else:
                        break
                
                # If we have multiple items or a single item not in a proper list, create a group
                if len(consecutive_items) > 1 or not isinstance(item.parent.resolve(doc) if item.parent else None, (OrderedList, UnorderedList)):
                    # Create appropriate list group
                    if list_type == GroupLabel.ORDERED_LIST:
                        list_group = doc.add_ordered_list(parent=item.parent.resolve(doc) if item.parent else None)
                    else:
                        list_group = doc.add_unordered_list(parent=item.parent.resolve(doc) if item.parent else None)
                    
                    # Move all consecutive items to the new group
                    for list_item in consecutive_items:
                        doc.insert_item_before_sibling(new_item=doc.add_list_item(
                            text=list_item.text,
                            enumerated=list_item.enumerated,
                            marker=list_item.marker,
                            orig=list_item.orig,
                            prov=list_item.prov[0] if list_item.prov else None,
                            parent=list_group,
                            content_layer=list_item.content_layer,
                            formatting=list_item.formatting,
                            hyperlink=list_item.hyperlink
                        ), sibling=list_item)
                    
                    # Delete original items
                    doc.delete_items(node_items=consecutive_items)
                
                i = j
            else:
                i += 1
    
    def process_document(self, doc: DoclingDocument) -> DoclingDocument:
        """
        Process the entire document to identify and convert list markers.
        
        Args:
            doc: The DoclingDocument to process
            
        Returns:
            The processed document (modified in-place)
        """
        # Find all marker-content pairs
        pairs = self.find_marker_content_pairs(doc)
        
        # Process pairs in reverse order to avoid index issues when deleting items
        for marker_item, content_item in reversed(pairs):
            # Create new ListItem
            list_item = self.create_list_item_from_pair(marker_item, content_item)
            
            # Insert the new ListItem
            doc.insert_item_before_sibling(new_item=list_item, sibling=marker_item)
            
            # Delete original items
            items_to_delete = [marker_item]
            if content_item:
                items_to_delete.append(content_item)
            doc.delete_items(node_items=items_to_delete)
        
        # Group consecutive list items
        self.group_consecutive_list_items(doc)
        
        return doc


# Example usage and testing
def example_usage():
    """Example of how to use the ListItemMarkerProcessor."""
    from docling_core.types.doc.document import DoclingDocument, ProvenanceItem
    from docling_core.types.doc.base import BoundingBox, CoordOrigin
    
    # Create a sample document
    doc = DoclingDocument(name="Sample Document")
    
    # Add some sample text items that should be converted to list items
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="1.",  # Marker only
        prov=ProvenanceItem(
            page_no=0,
            bbox=BoundingBox(l=0, t=0, r=10, b=10, coord_origin=CoordOrigin.TOPLEFT),
            charspan=(0, 2)
        )
    )
    
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="First item content",  # Content only
        prov=ProvenanceItem(
            page_no=0,
            bbox=BoundingBox(l=15, t=0, r=100, b=10, coord_origin=CoordOrigin.TOPLEFT),
            charspan=(0, 18)
        )
    )
    
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="• Second item with bullet and content",  # Marker and content together
        prov=ProvenanceItem(
            page_no=0,
            bbox=BoundingBox(l=0, t=15, r=200, b=25, coord_origin=CoordOrigin.TOPLEFT),
            charspan=(0, 37)
        )
    )
    
    # Process the document
    processor = ListItemMarkerProcessor()
    processed_doc = processor.process_document(doc)
    
    return processed_doc


if __name__ == "__main__":
    # Run example
    doc = example_usage()
    print("Document processed successfully!")
    print(f"Number of texts: {len(doc.texts)}")
    print(f"Number of groups: {len(doc.groups)}")
