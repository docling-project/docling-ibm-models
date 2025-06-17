from docling_core.types.doc.document import DoclingDocument, ProvenanceItem
from docling_core.types.doc.base import BoundingBox, CoordOrigin

from docling_ibm_models.listitem_normalizer.list_marker_processor import ListItemMarkerProcessor

# Example usage and testing
def test_listitem_marker_model():
    """Example of how to use the ListItemMarkerProcessor."""
    
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
    
    assert len(processed_doc.texts)==2, "len(processed_doc.texts)==2"

    assert processed_doc.texts[0].text=="First item content"
    assert processed_doc.texts[0].marker=="1."

    assert processed_doc.texts[1].text=="Second item with bullet and content"
    assert processed_doc.texts[1].marker=="•"
