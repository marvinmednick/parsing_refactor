from typing import Optional, List, Tuple, Any
from .models import StructuredSection

# This module is now less central but can hold basic helpers if needed.
# The main prediction logic is in SectionProcessor.
# We might still need helpers for incrementing values.


class SectionOperationsHelper:
    """Holds static helper methods useful for section logic."""

    @staticmethod
    def increment_value(value: Any) -> Optional[Any]:
        """Increments numeric or single alpha character values."""
        if isinstance(value, int):
            return value + 1
        elif isinstance(value, str) and len(value) == 1 and "A" <= value.upper() <= "Z":
            char_code = ord(value)
            if value.isupper():
                return (
                    chr(char_code + 1) if value != "Z" else None
                )  # Simple overflow check
            else:
                return (
                    chr(char_code + 1) if value != "z" else None
                )  # Simple overflow check
        # Add support for other increments if needed (e.g., Roman numerals?)
        return None  # Cannot increment other types

    @staticmethod
    def get_structure_string(structure: StructuredSection) -> str:
        """Converts a structured section back to a string representation (e.g., "Annex A.1")."""
        if not structure:
            return ""
        parts = []
        if structure[0][0] == "P":  # Handle Prefix
            parts.append(str(structure[0][1]))
            structure_to_join = structure[1:]
        else:
            structure_to_join = structure

        string_parts = [str(item[1]) for item in structure_to_join]
        id_part = ".".join(string_parts)

        if parts:  # If prefix exists
            return f"{parts[0]} {id_part}"
        else:
            return id_part
