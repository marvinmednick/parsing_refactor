import re
from typing import Dict, Optional, Pattern, Any
from dataclasses import dataclass, field


@dataclass
class CategoryDebugConfig:
    """Configuration for a single debug category."""

    name: str
    start_line: Optional[int] = None
    stop_line: Optional[int] = None
    start_pattern: Optional[str] = None
    stop_pattern: Optional[str] = None
    initially_active: bool = False
    # Internal state
    start_regex: Optional[Pattern] = field(init=False, default=None)
    stop_regex: Optional[Pattern] = field(init=False, default=None)
    is_active: bool = field(init=False, default=False)
    # Track if regex activation occurred to override line numbers
    regex_activated: bool = field(init=False, default=False)
    regex_deactivated: bool = field(init=False, default=False)

    def __post_init__(self):
        """Compile regex patterns after initialization."""
        if self.start_pattern:
            try:
                self.start_regex = re.compile(self.start_pattern)
            except re.error as e:
                print(
                    f"Warning: Invalid start_pattern regex for category '{self.name}': {e}"
                )
        if self.stop_pattern:
            try:
                self.stop_regex = re.compile(self.stop_pattern)
            except re.error as e:
                print(
                    f"Warning: Invalid stop_pattern regex for category '{self.name}': {e}"
                )
        self.is_active = self.initially_active
        self.regex_activated = False
        self.regex_deactivated = False


class DebugManager:
    """Manages debug categories and printing based on configuration."""

    def __init__(self):
        self.categories: Dict[str, CategoryDebugConfig] = {}
        self.current_line_num: int = -1

    def configure(self, category_configs: Dict[str, CategoryDebugConfig]):
        """Applies loaded debug configurations."""
        self.categories = category_configs
        # Reset state if re-configuring
        self.current_line_num = -1
        for cat in self.categories.values():
            cat.is_active = cat.initially_active
            cat.regex_activated = False
            cat.regex_deactivated = False

    def process_line(self, line_num: int, line_text: str):
        """Updates the active state of debug categories based on the current line."""
        self.current_line_num = line_num

        for category in self.categories.values():
            # --- Activation Logic ---
            activated_this_line = False
            # Regex activation
            if (
                category.start_regex
                and not category.is_active
                and not category.regex_activated
            ):
                if category.start_regex.search(line_text):
                    category.is_active = True
                    category.regex_activated = True  # Mark that regex activated it
                    activated_this_line = True
                    print(
                        f"[DEBUG Event] Category '{category.name}' activated by regex at line {line_num}"
                    )

            # Line number activation (only if not already regex-activated)
            if (
                not activated_this_line
                and category.start_line is not None
                and not category.is_active
                and not category.regex_activated
                and line_num >= category.start_line
            ):
                category.is_active = True
                activated_this_line = True
                print(
                    f"[DEBUG Event] Category '{category.name}' activated by line number {line_num} (>= {category.start_line})"
                )

            # --- Deactivation Logic ---
            deactivated_this_line = False
            # Regex deactivation (takes precedence if active)
            if (
                category.stop_regex
                and category.is_active
                and not category.regex_deactivated
            ):
                if category.stop_regex.search(line_text):
                    category.is_active = False
                    category.regex_deactivated = True  # Mark that regex deactivated it
                    deactivated_this_line = True
                    print(
                        f"[DEBUG Event] Category '{category.name}' deactivated by regex at line {line_num}"
                    )

            # Line number deactivation (only if active and not just regex-deactivated)
            if (
                not deactivated_this_line
                and category.stop_line is not None
                and category.is_active
                and
                # Don't deactivate by line# if regex is keeping it active OR if regex deactivated it
                not category.regex_activated
                and not category.regex_deactivated
                and line_num >= category.stop_line
            ):
                category.is_active = False
                deactivated_this_line = True
                print(
                    f"[DEBUG Event] Category '{category.name}' deactivated by line number {line_num} (>= {category.stop_line})"
                )

    def is_active(self, category_key: str) -> bool:
        """Checks if a specific debug category is currently active."""
        category = self.categories.get(category_key)
        return category.is_active if category else False

    def debug(self, category_key: str, *args: Any, **kwargs: Any):
        """Prints debug information if the category is active."""
        if self.is_active(category_key):
            message_parts = [str(arg) for arg in args]
            message_parts.extend([f"{k}={v}" for k, v in kwargs.items()])
            prefix = f"DEBUG [{category_key} L:{self.current_line_num + 1}]:"  # Use 1-based line num for display
            print(prefix, " ".join(message_parts))
