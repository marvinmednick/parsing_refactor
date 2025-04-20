# src/pdf_parser/utils.py

from typing import Optional, List, Tuple, Any, Dict
from .models import StructuredSection

# Import SectionHandler type hint - avoid full import if possible to prevent cycles
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .handlers import SectionHandler  # Use TYPE_CHECKING block


# --- compare_structure function (from Response #79) ---
def compare_structure(struct_a: Optional[List[Tuple[str, Any]]], struct_b: Optional[List[List[Any]]], debug: bool = False) -> Tuple[bool, Optional[str]]:
    """Compares two structure representations."""
    # ... (Full implementation as provided in Response #79) ...
    reason: Optional[str] = None
    if struct_b is None:
        return (True, None)
    if not isinstance(struct_a, list):
        reason = f"Type mismatch: Struct A is not list ({type(struct_a)})"
    elif not isinstance(struct_b, list):
        reason = f"Type mismatch: Struct B is not list ({type(struct_b)})"
    elif len(struct_a) != len(struct_b):
        reason = f"Length mismatch: A={len(struct_a)}, B={len(struct_b)}"
    else:
        for i in range(len(struct_a)):
            el_a, el_b = struct_a[i], struct_b[i]
            if not isinstance(el_a, tuple) or len(el_a) != 2:
                reason = f"Format A[{i}]: {el_a}"
                break
            if not isinstance(el_b, list) or len(el_b) != 2:
                reason = f"Format B[{i}]: {el_b}"
                break
            type_a, value_a = el_a
            type_b, value_b = el_b
            if type_a != type_b:
                reason = f"Type mismatch[{i}]: A='{type_a}', B='{type_b}'"
                break
            if value_a != value_b:
                reason = f"Value mismatch[{i}]: A='{value_a}', B='{value_b}'"
                break
    if reason:
        if debug:
            print(f"DEBUG compare_structure: {reason}")
        return (False, reason)
    else:
        return (True, None)


# --- NEW function to parse identifier strings ---
def parse_identifier_string(
    identifier_str: str,
    handlers: Dict[str, "SectionHandler"],  # Pass available handlers
    handler_check_order: List[str],  # Pass preferred check order
) -> Optional[StructuredSection]:
    """
    Parses a section identifier string (e.g., "Annex A", "1.2.1") into a
    StructuredSection using the provided handlers.

    Args:
        identifier_str: The string to parse (should not include the title).
        handlers: Dictionary mapping type names to initialized handler instances.
        handler_check_order: Preferred order to check handlers.

    Returns:
        The parsed StructuredSection list, or None if parsing fails.
    """
    if not identifier_str or not isinstance(identifier_str, str):
        return None

    text_to_parse = identifier_str.strip()
    # Construct a fake line matching handler regex format (ID + Title)
    # Needed because handlers expect text following the ID
    fake_line = f"{text_to_parse} Placeholder Title"

    # Create ordered list of handlers to check
    handlers_to_check = []
    available_handlers = list(handlers.keys())
    for name in handler_check_order:
        if name in handlers:
            handlers_to_check.append(handlers[name])
            if name in available_handlers:
                available_handlers.remove(name)
    for name in available_handlers:
        handlers_to_check.append(handlers[name])

    # Try parsing with handlers in order
    for handler in handlers_to_check:
        # Use match on the fake line to simulate finding a header start
        match_obj = handler.get_regex().match(fake_line)
        if match_obj:
            # Pass the original fake_line to parse, as it expects the full match
            parsed = handler.parse(fake_line)
            if parsed:
                return parsed  # Return first successful parse

    # If no handler could parse it
    print(f"Warning: parse_identifier_string failed for input '{identifier_str}'")  # Add warning
    return None

