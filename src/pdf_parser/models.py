from typing import List, Tuple, Any, Dict, TypedDict

# Type alias for the structured representation of a section identifier
StructuredSection = List[Tuple[str, Any]]

# Type alias for an element on the hierarchy stack (could be a class/dataclass too)
StackElement = TypedDict(
    "StackElement",
    {
        "S": StructuredSection,  # Structure
        "T": str,  # Type Name
        "text": str,  # Original header text
        "line_idx": int,  # Line index where header started
    },
)


# Structure for the final output section
class OutputSection(TypedDict):
    section_title: str
    section_text: str
    page_number: int
    images: List[Dict[str, Any]]  # List of image info dicts
    # Include structure info if desired
    structure: StructuredSection
    type: str
