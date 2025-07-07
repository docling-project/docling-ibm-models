from typing import Dict


class LayoutLabels:
    r"""Single point of reference for the layout labels"""

    def __init__(self) -> None:
        r""" """
        # Canonical classes originating in DLNv2
        self._canonical: Dict[int, str] = {
            # DLNv1 and DLNv2
            0: "Caption",
            1: "Footnote",
            2: "Formula",
            3: "List-item",
            4: "Page-footer",
            5: "Page-header",
            6: "Picture",
            7: "Section-header",
            8: "Table",
            9: "Text",
            10: "Title",
            # DLNv2 only
            11: "Document Index",
            12: "Code",
            13: "Checkbox-Selected",
            14: "Checkbox-Unselected",
            15: "Form",
            16: "Key-Value Region",
        }
        self._inverse_canonical: Dict[str, int] = {
            label: class_id for class_id, label in self._canonical.items()
        }

        # Shifted canonical classes with background in 0
        self._shifted_canonical: Dict[int, str] = {0: "Background"}
        for k, v in self._canonical.items():
            self._shifted_canonical[k + 1] = v
        self._inverse_shifted_canonical: Dict[str, int] = {
            label: class_id for class_id, label in self._shifted_canonical.items()
        }

    def canonical_categories(self) -> Dict[int, str]:
        return self._canonical

    def canonical_to_int(self) -> Dict[str, int]:
        return self._inverse_canonical

    def shifted_canonical_categories(self) -> Dict[int, str]:
        return self._shifted_canonical

    def shifted_canonical_to_int(self) -> Dict[str, int]:
        return self._inverse_shifted_canonical
