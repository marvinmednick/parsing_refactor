# src/pdf_parser/handlers.py
# Refactored handlers: Numeric and AlphaNumeric now handle optional prefixes/suffixes

import re
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Any, Pattern
from .models import StructuredSection  # Use models for types
# Assuming DebugManager could be used for parse errors, but need instance.
# For simplicity, parse errors will print for now or return None.
# from .debug import DebugManager

# --- Section Handler Framework ---


class SectionHandler(ABC):
    """Abstract Base Class for section type handlers."""

    # Make type_name an instance variable set in __init__
    type_name: str
    prefix: Optional[str]
    allow_trailing_dot: bool
    REGEX: Pattern  # Compiled regex stored here

    @abstractmethod
    def get_type_name(self) -> str:
        """Returns the unique name of this section type instance."""
        pass

    @abstractmethod
    def get_regex(self) -> Pattern:
        """Returns the compiled regex used by this handler instance."""
        pass

    @abstractmethod
    def parse(self, text: str) -> Optional[StructuredSection]:
        """
        Parses the relevant part of the text into the standardized structured list format.
        Returns None if the text doesn't strictly conform after regex matching.
        """
        pass

    @abstractmethod
    def get_default_start_structure(self) -> Optional[StructuredSection]:
        """
        Returns the default starting structure for this type when transitioned to.
        Returns None if a default start isn't applicable or requires context.
        """
        pass


class NumericHandler(SectionHandler):
    """
    Handles purely numeric sections like 1, 1.2, 1.2.3.
    Can be initialized with an optional prefix (e.g., "Appendix") and allow a trailing dot.
    """

    CORE_ID_PATTERN = r"\d+(?:\.\d+)*"  # Class constant for the numeric ID part

    def __init__(
        self,
        type_name: str = "Numeric",
        prefix: Optional[str] = None,
        allow_trailing_dot: bool = False,
    ):
        """
        Initializes the handler.

        Args:
            type_name: The unique name for this specific handler instance (e.g., "Numeric", "AppendixNumeric").
            prefix: Optional string prefix (e.g., "Appendix").
            allow_trailing_dot: If True, allows an optional dot after the number (e.g., "Appendix 1.").
        """
        self.type_name = type_name
        self.prefix = prefix.strip() if prefix else None
        self.allow_trailing_dot = allow_trailing_dot
        self.REGEX = self._build_regex()
        print(
            f"DEBUG Handler Init: Name='{self.type_name}', Prefix='{self.prefix}', AllowDot={self.allow_trailing_dot}, Regex='{self.REGEX.pattern}'"
        )  # Debug Init

    def _build_regex(self) -> Pattern:
        """Builds the regex dynamically based on prefix and suffix options."""
        prefix_pattern = rf"^{re.escape(self.prefix)}\s+" if self.prefix else "^"
        id_pattern = f"({self.CORE_ID_PATTERN})"  # Capture group 1 for the ID
        suffix_pattern = r"\.?" if self.allow_trailing_dot else ""
        title_pattern = r"\s+(.+)$"  # Capture group 2 for the title

        full_pattern = f"{prefix_pattern}{id_pattern}{suffix_pattern}{title_pattern}"
        return re.compile(
            full_pattern, re.IGNORECASE if self.prefix else 0
        )  # Ignore case only if prefix exists?

    def get_type_name(self) -> str:
        return self.type_name

    def get_regex(self) -> Pattern:
        return self.REGEX

    def parse(self, text: str) -> Optional[StructuredSection]:
        match = self.get_regex().match(text.strip())
        if not match:
            return None

        section_id_part = match.group(1)  # Group 1 captured the core ID pattern

        try:
            # Convert "1.2.3" into [('N', 1), ('N', 2), ('N', 3)]
            base_structure = [("N", int(p)) for p in section_id_part.split(".")]
        except ValueError:
            # This really shouldn't happen if the number regex part worked
            print(
                f"ERROR: NumericHandler '{self.type_name}' failed numeric conversion for matched ID '{section_id_part}' in text '{text}'"
            )
            return None

        # Prepend prefix if this handler instance has one
        if self.prefix:
            return [("P", self.prefix)] + base_structure
        else:
            return base_structure

    def get_default_start_structure(self) -> Optional[StructuredSection]:
        # Base default is section "1"
        base_start: StructuredSection = [("N", 1)]
        # Prepend prefix if this handler instance has one
        if self.prefix:
            return [("P", self.prefix)] + base_start
        else:
            return base_start


class AlphaNumericHandler(SectionHandler):
    """
    Handles sections starting with Alpha, then Numeric like A, A.1, B.2.1.
    Can be initialized with an optional prefix (e.g., "Annex") and allow a trailing dot.
    """

    # Core pattern matches single Alpha, then optional ".N..."
    CORE_ID_PATTERN = r"[A-Za-z](?:\.\d+)*"

    def __init__(
        self,
        type_name: str = "AlphaNumeric",
        prefix: Optional[str] = None,
        allow_trailing_dot: bool = False,
    ):
        """
        Initializes the handler.

        Args:
            type_name: The unique name for this specific handler instance (e.g., "AlphaNumeric", "Annex").
            prefix: Optional string prefix (e.g., "Annex").
            allow_trailing_dot: If True, allows an optional dot after the ID (e.g., "Annex A.").
        """
        self.type_name = type_name
        self.prefix = prefix.strip() if prefix else None
        self.allow_trailing_dot = allow_trailing_dot
        self.REGEX = self._build_regex()
        print(
            f"DEBUG Handler Init: Name='{self.type_name}', Prefix='{self.prefix}', AllowDot={self.allow_trailing_dot}, Regex='{self.REGEX.pattern}'"
        )  # Debug Init

    def _build_regex(self) -> Pattern:
        """Builds the regex dynamically based on prefix and suffix options."""
        prefix_pattern = rf"^{re.escape(self.prefix)}\s+" if self.prefix else "^"
        id_pattern = f"({self.CORE_ID_PATTERN})"  # Capture group 1 for the ID
        suffix_pattern = r"\.?" if self.allow_trailing_dot else ""
        title_pattern = r"\s+(.+)$"  # Capture group 2 for the title

        full_pattern = f"{prefix_pattern}{id_pattern}{suffix_pattern}{title_pattern}"
        return re.compile(full_pattern, re.IGNORECASE if self.prefix else 0)

    def get_type_name(self) -> str:
        return self.type_name

    def get_regex(self) -> Pattern:
        return self.REGEX

    def parse(self, text: str) -> Optional[StructuredSection]:
        match = self.get_regex().match(text.strip())
        if not match:
            return None

        section_id_part = match.group(1)  # Group 1 captured the core ID pattern
        parts = section_id_part.split(".")
        structure: StructuredSection = []

        try:
            # First part must be a single letter
            if len(parts[0]) == 1 and parts[0].isalpha():
                structure.append(("A", parts[0]))  # Use 'A' for Alpha component type
            else:
                # Should not be reached if regex is correct
                print(
                    f"ERROR: AlphaNumericHandler '{self.type_name}' parse expected single letter start, got: {parts[0]} in text '{text}'"
                )
                return None

            # Subsequent parts must be numeric
            for part in parts[1:]:
                if part.isdigit():
                    structure.append(
                        ("N", int(part))
                    )  # Use 'N' for Numeric component type
                else:
                    print(
                        f"ERROR: AlphaNumericHandler '{self.type_name}' parse expected numeric subsection, got: {part} in text '{text}'"
                    )
                    return None
        except (ValueError, IndexError) as e:
            # Should not happen if regex works
            print(
                f"ERROR: AlphaNumericHandler '{self.type_name}' parse failed for ID '{section_id_part}' in text '{text}': {e}"
            )
            return None

        # Prepend prefix if this handler instance has one
        if self.prefix:
            return [("P", self.prefix)] + structure
        else:
            return structure

    def get_default_start_structure(self) -> Optional[StructuredSection]:
        # Base default is section "A"
        base_start: StructuredSection = [("A", "A")]
        # Prepend prefix if this handler instance has one
        if self.prefix:
            return [("P", self.prefix)] + base_start
        else:
            return base_start
