# src/pdf_parser/state.py
# Refactored SectionProcessor using current_section_info + prediction lists/stack

import re
import langdetect
from typing import Dict, Optional, List, Tuple, Any, Set

# Assuming these modules exist in the same package
from .models import StructuredSection, StackElement, OutputSection
from .handlers import SectionHandler  # Keep import for type hints
from .operations import SectionOperationsHelper  # Import helpers
from .debug import DebugManager
from .exceptions import ParsingLogicError, ConfigError
from .utils import compare_structure, parse_identifier_string


# Define types for prediction structures for clarity
NextSibPrediction = Optional[StructuredSection]
SubSecPredictions = List[StructuredSection]  # List of possible first subsections
NewTypePredictions = List[StructuredSection]  # List of default starts for new types
# This stack holds predictions for NEXT SIBLINGS of PARENTS (using structure)
ParentSibPredictionStack = List[NextSibPrediction]  # Type alias

# --- Constants ---
# Consider moving these to config or a constants module
MIN_KEEP_LENGTH = 512

try:
    langdetect.DetectorFactory.seed = 42
except NameError:
    print("Warning: langdetect library not found or seeding failed.")
    pass


class SectionProcessor:
    def __init__(self, config: Dict, handlers: Dict[str, "SectionHandler"], debug_manager: DebugManager):  # Removed handler_min_levels if not using that approach
        self.config = config
        self.handlers = handlers
        self.debug = debug_manager
        self.rules = config.get("valid_next_section_rules", {})  # Keep rules dict
        self.policy = config.get("first_section_policy", {"accept_any": True})
        self.constants = config.get("constants", {})

        # --- State ---
        self.current_section_info: Optional[StackElement] = None
        self.current_section_text_lines: List[Tuple[int, str]] = []
        self.processed_sections: List[OutputSection] = []
        self.first_section_found: bool = False

        # --- Predictions ---
        self.predicted_next_sibling: NextSibPrediction = None
        self.predicted_subsections: SubSecPredictions = []
        self.predicted_new_types: NewTypePredictions = []
        self.parent_sibling_prediction_stack: ParentSibPredictionStack = []

        self.line_counter = 0

        # --- Post-process rules AFTER handlers are available ---
        # This modifies self.rules in place, converting strings to structures
        self._process_config_rules()

        self.debug.debug("state_init", f"SectionProcessor initialized. Handlers: {list(self.handlers.keys())}")
        # Initial prediction based on policy (and potentially processed rules)
        self._calculate_predictions()

    def _process_config_rules(self):
        """
        Iterates through loaded rules and parses any string-based
        required_start_structure values into lists of StructuredSection objects,
        storing them back into the self.rules dictionary.
        """
        self.debug.debug("config_proc", "Post-processing config rules for required_start_structure strings...")
        # Define order here or make it configurable/passed in
        handler_check_order = list(self.config.get("handler_check_order", ["Annex", "Appendix", "Numeric", "AlphaNumeric"]))

        if not self.rules:  # No rules defined
            self.debug.debug("config_proc", "No 'valid_next_section_rules' found in config.")
            return

        for type_name, rule_list in self.rules.items():
            if not isinstance(rule_list, list):
                continue
            for rule_idx, rule in enumerate(rule_list):  # Use enumerate for better error messages
                if not isinstance(rule, dict):
                    continue

                op = rule.get("operation")
                req_struct_val = rule.get("required_start_structure")

                # If it's a new_section_type rule with a required structure defined
                if op == "new_section_type" and req_struct_val:
                    parsed_struct_list: List[StructuredSection] = []  # Store results here
                    has_error = False
                    structure_strings_to_parse: List[str] = []

                    # Determine if it's a single string or a list of strings
                    if isinstance(req_struct_val, str):
                        structure_strings_to_parse = [req_struct_val]
                    elif isinstance(req_struct_val, list) and all(isinstance(s, str) for s in req_struct_val):
                        structure_strings_to_parse = req_struct_val
                    else:
                        # It's neither a string nor a list of strings, maybe already parsed?
                        # Or it's an invalid format. Check if it's already a list of lists (from JSON)
                        # For simplicity, assume it MUST be string or list of strings from YAML
                        if not isinstance(req_struct_val, list):  # It's not pre-parsed list and not string/list-of-strings
                            self.debug.debug(
                                "config_proc",
                                f"  WARNING: Invalid format for required_start_structure in rule #{rule_idx + 1} for '{type_name}'. Expected string or list of strings. Value: {req_struct_val}",
                            )
                            has_error = True
                        # If it *is* already a list, assume it might be pre-parsed list-of-tuples (though config should use strings)
                        # Or it might be list-of-lists from JSON format if user didn't update config fully
                        elif isinstance(req_struct_val, list):
                            # Let calculate_predictions handle it if it's already processed structure
                            self.debug.debug("config_proc", f"  Found existing list for required_start_structure rule #{rule_idx + 1} in '{type_name}'. Assuming pre-parsed or will be handled later.")
                            continue  # Skip processing here

                    # Parse each string in the list
                    for structure_string in structure_strings_to_parse:
                        self.debug.debug("config_proc", f"  Parsing required_start_structure string '{structure_string}' for rule in '{type_name}'.")
                        # Use the utility function
                        parsed_struct = parse_identifier_string(structure_string, self.handlers, handler_check_order)
                        if parsed_struct:
                            parsed_struct_list.append(parsed_struct)
                            self.debug.debug("config_proc", f"    Successfully parsed to: {parsed_struct}")
                        else:
                            self.debug.debug("config_proc", f"    WARNING: Failed to parse required_start_structure string '{structure_string}'. It will be ignored for this rule.")
                            has_error = True

                    # Store the list of successfully parsed structures back into the rule
                    rule["required_start_structure"] = parsed_struct_list  # Store list of structures
                    if not parsed_struct_list and not has_error and structure_strings_to_parse:
                        self.debug.debug(
                            "config_proc", f"  WARNING: required_start_structure parsing resulted in empty list for rule #{rule_idx + 1} in '{type_name}'. Input: {structure_strings_to_parse}"
                        )

    def _calculate_predictions(self):
        """
        Calculates predictions, using pre-parsed required_start_structure from rules.
        """
        current_section_info = self.current_section_info
        current_struct_str = SectionOperationsHelper.get_structure_string(current_section_info["S"]) if current_section_info else "None"
        self.debug.debug("prediction", f"Calculating predictions based on current: {current_struct_str}")

        # Reset predictions
        self.predicted_next_sibling = None
        self.predicted_subsections = []
        self.predicted_new_types = []

        # --- Predict First Section ---
        if not current_section_info:
            if not self.policy.get("accept_any", False):
                must_struct_config = self.policy.get("must_start_with_structure")  # Expects list of lists e.g. [['N',1]]
                if must_struct_config and isinstance(must_struct_config, list):
                    try:
                        self.predicted_next_sibling = [tuple(item) for item in must_struct_config]  # Convert to list of tuples
                        self.debug.debug("prediction", f"Predicting REQUIRED first section based on policy: {self.predicted_next_sibling}")
                    except Exception as e:
                        self.debug.debug("prediction", f"Error creating first section prediction from policy: {e}")
                        self.predicted_next_sibling = None
            self.debug.debug("prediction", f"Initial state predictions: Required First={self.predicted_next_sibling}, Subsections=[], New Types=[]")
            self.debug.debug("prediction", f"--- End Predictions (Initial State) ---")
            return

        # --- Predict based on Current Section ---
        current_structure = current_section_info["S"]
        current_type = current_section_info["T"]

        # 1. Predict Next Sibling (Conventional)
        # ... (logic remains the same) ...
        if current_structure:
            last_type, last_val = current_structure[-1]
            next_val = SectionOperationsHelper.increment_value(last_val)
            if next_val is not None:
                self.predicted_next_sibling = current_structure[:-1] + [(last_type, next_val)]
        self.debug.debug("prediction", f"Predicted next sibling: {SectionOperationsHelper.get_structure_string(self.predicted_next_sibling) if self.predicted_next_sibling else 'None'}")

        # --- Predict Subsections (Including Type Changes/Prefix Drops) ---
        # Reset list for this calculation
        self.predicted_subsections = []

        # 2a. Predict conventional subsections (keeping prefix)
        # Using convention: numeric '1' and alpha 'A'.
        # TODO: Make conventional prediction configurable if needed
        numeric_subsection_struct = current_structure + [("N", 1)]
        self.predicted_subsections.append(numeric_subsection_struct)
        if current_structure and current_structure[-1][0] != "A":  # Avoid A -> A.A ? Convention check.
            alpha_subsection_struct = current_structure + [("A", "A")]
            self.predicted_subsections.append(alpha_subsection_struct)
        self.debug.debug("prediction", f"  Added conventional subsection predictions: {[SectionOperationsHelper.get_structure_string(s) for s in self.predicted_subsections]}")

        # 2b. Predict subsections based on 'new_subsection_type' rules in config (dropping prefix)
        allowed_rules = self.rules.get(current_type, [])
        sub_type_rules = [rule for rule in allowed_rules if rule.get("operation") == "new_subsection_type"]
        self.debug.debug("prediction", f"  Found {len(sub_type_rules)} 'new_subsection_type' rule(s) for type '{current_type}'")

        # --- Add detailed loop debug from Response #85 ---
        for rule_idx, rule in enumerate(sub_type_rules):
            self.debug.debug("prediction_detail", f"  Processing sub_type_rule #{rule_idx + 1}: {rule}")
            params = rule.get("params")
            target_type = rule.get("target_type")
            if not params or not target_type:
                self.debug.debug("prediction_detail", "    Rule missing params or target_type, skipping.")
                continue

            # Extract core structure
            core_structure = list(current_structure)
            prefix_val = None
            if core_structure and core_structure[0][0] == "P":
                prefix_val = core_structure[0][1]
                core_structure = core_structure[1:]
            self.debug.debug("prediction_detail", f"    Extracted core_structure: {core_structure} (Prefix was: {prefix_val})")

            if not core_structure:
                self.debug.debug("prediction_detail", "    Core structure empty, skipping rule.")
                continue

            # Generate subsection based on params
            allowed_types = params.get("allowed_types")
            start_value = params.get("start_with")
            if not allowed_types or start_value is None:
                self.debug.debug("prediction_detail", f"    Rule missing allowed_types or start_with in params, skipping: {params}")
                continue

            new_level_type_config = allowed_types[0]
            new_comp_type = "N" if new_level_type_config == "Numeric" else "A" if new_level_type_config == "Alpha" else None
            self.debug.debug("prediction_detail", f"    Rule params indicate new component type '{new_comp_type}' with value '{start_value}'")

            if new_comp_type:
                # Build structure WITHOUT prefix
                new_structure_no_prefix = core_structure + [(new_comp_type, start_value)]
                self.debug.debug("prediction_detail", f"    Generated no-prefix structure: {new_structure_no_prefix}")
                # Check if this prediction already exists before adding
                if new_structure_no_prefix not in self.predicted_subsections:
                    self.predicted_subsections.append(new_structure_no_prefix)
                    self.debug.debug("prediction_detail", f"    **ADDED** subsection prediction (from rule, prefix dropped): {new_structure_no_prefix}")
                else:
                    self.debug.debug("prediction_detail", f"    Skipping duplicate subsection prediction (from rule): {new_structure_no_prefix}")
            else:
                self.debug.debug("prediction_detail", f"    Could not determine valid new component type from params, skipping rule: {params}")
        # --- End detailed loop debug ---

        # Log final list of ALL predicted subsections
        self.debug.debug("prediction", f"Final Predicted subsections: {[SectionOperationsHelper.get_structure_string(s) for s in self.predicted_subsections]}")

        # 3. Predict New Section Types (Using pre-parsed rules)
        # This block should come AFTER subsection prediction
        self.predicted_new_types = []  # Reset list
        allowed_rules = self.rules.get(current_type, [])
        transition_rules = [rule for rule in allowed_rules if rule.get("operation") == "new_section_type"]
        self.debug.debug("prediction", f"Found {len(transition_rules)} 'new_section_type' rule(s) for type '{current_type}'")

        processed_targets = set()  # Avoid adding default if specific rule existed

        for rule in transition_rules:
            target_type_name = rule.get("target_type")
            if not target_type_name:
                continue

            # Check for pre-parsed required_start_structure LIST
            required_starts: Optional[List[StructuredSection]] = rule.get("required_start_structure")

            if isinstance(required_starts, list):  # Check if it's a list (should be pre-parsed)
                if required_starts:  # List is not empty
                    self.debug.debug("prediction", f"  Using specific required_start_structure list for '{target_type_name}' from rule: {required_starts}")
                    for struct in required_starts:
                        if struct not in self.predicted_new_types:
                            self.predicted_new_types.append(struct)
                else:  # List is empty (means parsing failed or explicitly empty)
                    self.debug.debug("prediction", f"  Rule for '{target_type_name}' provided empty required_start_structure list. No prediction added from this rule.")
                processed_targets.add(target_type_name)  # Mark as handled by this rule

            # If required_start_structure was NOT defined or not a list, use default
            elif target_type_name not in processed_targets:
                target_handler = self.handlers.get(target_type_name)
                if target_handler:
                    default_start = target_handler.get_default_start_structure()
                    if default_start:
                        self.debug.debug("prediction", f"  Using default start structure for '{target_type_name}': {default_start}")
                        if default_start not in self.predicted_new_types:
                            self.predicted_new_types.append(default_start)
                    else:
                        self.debug.debug("prediction", f"  Target handler '{target_type_name}' has no default start structure.")
                else:
                    self.debug.debug("prediction", f"  Target handler '{target_type_name}' not found.")
                processed_targets.add(target_type_name)  # Mark target as processed (by default)

        self.debug.debug("prediction", f"Final Predicted new types: {[SectionOperationsHelper.get_structure_string(s) for s in self.predicted_new_types]}")
        self.debug.debug("prediction", f"--- End Predictions for {current_struct_str} ---")

    def _perform_language_check(self, text: str) -> bool:
        """Checks if text is likely English. Returns True if OK, False otherwise."""
        min_keep = self.constants.get("MIN_KEEP_LENGTH", MIN_KEEP_LENGTH)
        if len(text) < min_keep:
            self.debug.debug(
                "lang_check",
                f"Skipping language check for short text (len {len(text)} < {min_keep})",
            )
            return True
        sample = text[:50000]
        try:
            lang = langdetect.detect(sample)
            self.debug.debug("lang_check", f"Detected language: {lang}")
            if lang != "en":
                self.debug.debug("lang_check", f"Language '{lang}' is not 'en'. Filtering section.")
                return False
        except langdetect.lang_detect_exception.LangDetectException:
            self.debug.debug("lang_check", "Language detection failed. Assuming OK.")
            pass
        except Exception as e:
            self.debug.debug("lang_check", f"ERROR during language detection: {e}")
            pass
        return True

    def _finalize_current_section(self, end_line_idx: int):
        """Adds the completed current section to the results list."""
        if not self.current_section_info:
            self.debug.debug("finalize", "Called with no current section info. Nothing to finalize.")
            return

        current_struct_str = SectionOperationsHelper.get_structure_string(self.current_section_info["S"])
        self.debug.debug(
            "finalize",
            f"Finalizing section: '{self.current_section_info['text']}' ({current_struct_str}) ending before line {end_line_idx + 1}",
        )

        body_text = self.join_lines_internal(self.current_section_text_lines)
        self.debug.debug("finalize", f"Collected body text length: {len(body_text)}")

        is_lang_ok = self._perform_language_check(body_text)
        self.debug.debug("lang_check", f"Language check result: {is_lang_ok}")

        if not is_lang_ok:
            self.debug.debug(
                "finalize",
                f"Section '{self.current_section_info['text']}' filtered out by language check.",
            )
            # Clear text buffer, but KEEP current_section_info state until replaced
            self.current_section_text_lines = []
            return

        # --- Create and Store Section ---
        start_page = -1  # Placeholder
        start_line_absolute_idx = self.current_section_info["line_idx"]
        # TODO: Implement proper start page calculation
        self.debug.debug(
            "finalize",
            f"Start page number calculation needed (using placeholder {start_page}).",
        )

        output_sec: OutputSection = {
            "section_title": self.current_section_info["text"],
            "section_text": body_text,
            "page_number": start_page,
            "images": [],  # Placeholder
            "structure": self.current_section_info["S"],
            "type": self.current_section_info["T"],
        }

        self.debug.debug(
            "finalize",
            f"Attempting to append section '{output_sec['section_title']}'. Current processed_sections count: {len(self.processed_sections)}",
        )
        self.processed_sections.append(output_sec)
        last_title = self.processed_sections[-1]["section_title"] if self.processed_sections else "N/A"
        self.debug.debug(
            "finalize",
            f"Append successful? New processed_sections count: {len(self.processed_sections)}. Last added title: {last_title}",
        )

        # DO NOT reset current_section_info here. Reset text buffer only.
        self.current_section_text_lines = []

    def join_lines_internal(self, lines_info: List[Tuple[int, str]]) -> str:
        """Joins lines into a single string."""
        output = "\n".join([line for _, line in lines_info])
        return output.strip()

    # --- Public Methods ---

    def process_line(self, line_info: Tuple[int, str], line_idx: int):
        """
        Processes a single line, checking against predictions, updating state,
        and collecting text content using the minimal state approach.
        """
        self.line_counter = line_idx
        self.debug.process_line(line_idx, line_info[1])

        page_num, original_line_text = line_info
        line_text = original_line_text.strip()

        self.debug.debug("line_proc", f"Processing Line {line_idx + 1}: '{line_text[:80]}...'")

        # --- Skip blank lines / Configurable TOC lines ---
        max_dots = self.constants.get("MAX_DOTS", 16)
        if not line_text or line_text.count(".") >= max_dots:
            if self.current_section_info:  # Only add if a section is active
                self.current_section_text_lines.append((page_num, original_line_text))
                current_struct_str = SectionOperationsHelper.get_structure_string(self.current_section_info["S"])
                self.debug.debug(
                    "line_proc",
                    f"Line added as content (blank/dots) to section '{current_struct_str}'",
                )
            return

        # --- Try to parse the line as a potential header ---
        potential_structure: Optional[StructuredSection] = None
        potential_handler: Optional[SectionHandler] = None
        # Use the ordered handler check from previous refinement
        handler_check_order = [
            "Annex",
            "Appendix",
            "Numeric",
            "AlphaNumeric",
        ]  # TODO: Make configurable?
        handlers_to_check = []
        available_handlers = list(self.handlers.keys())
        for name in handler_check_order:
            if name in self.handlers:
                handlers_to_check.append(self.handlers[name])
                if name in available_handlers:
                    available_handlers.remove(name)
        for name in available_handlers:
            handlers_to_check.append(self.handlers[name])

        for handler in handlers_to_check:
            if handler.get_regex().match(line_text):
                self.debug.debug(
                    "line_proc",
                    f"Line matches regex for handler '{handler.get_type_name()}'",
                )
                parsed = handler.parse(line_text)
                if parsed:
                    potential_structure = parsed
                    potential_handler = handler
                    self.debug.debug(
                        "line_proc",
                        f"Line parsed by '{handler.get_type_name()}': {parsed}",
                    )
                    break

        # --- Line is NOT a potential header ---
        if potential_structure is None:
            if self.current_section_info:
                self.current_section_text_lines.append((page_num, original_line_text))
                current_struct_str = SectionOperationsHelper.get_structure_string(self.current_section_info["S"])
                self.debug.debug(
                    "line_proc",
                    f"Line added as content to section '{current_struct_str}'",
                )
            else:
                self.debug.debug(
                    "line_proc",
                    "Line is not a header and no current section active, discarding.",
                )
            return

        # --- Line IS a potential header: Check against predictions ---
        matched_prediction_type = None  # first, sibling, subsection, parent_sibling, new_section_type
        matched_parent_level = -1  # Index in parent prediction stack if matched

        potential_info: StackElement = {
            "S": potential_structure,
            "T": potential_handler.get_type_name(),
            "text": line_text,
            "line_idx": line_idx,
        }
        self.debug.debug(
            "line_proc",
            f"Potential Header Found: Type='{potential_info['T']}', Struct={potential_info['S']}",
        )

        # --- Determine Match Type ---
        if not self.first_section_found:
            # Check if it matches the required first section prediction
            policy_struct_prediction = self.predicted_next_sibling
            policy_struct_config_format = [list(t) for t in policy_struct_prediction] if policy_struct_prediction else None
            match_status, _ = compare_structure(potential_structure, policy_struct_config_format)

            if match_status:
                matched_prediction_type = "first"
                self.debug.debug("match", f"Matched First Section Policy: {potential_structure}")
            elif self.policy.get("accept_any", False):
                matched_prediction_type = "first"
                self.debug.debug(
                    "match",
                    f"Accepted First Section (accept_any=True): {potential_structure}",
                )
            # Else: No match, handled below

        else:  # Check subsequent sections
            sibling_match, _ = compare_structure(
                potential_structure,
                [list(t) for t in self.predicted_next_sibling] if self.predicted_next_sibling else None,
            )
            if sibling_match:
                matched_prediction_type = "sibling"
                self.debug.debug("match", f"Matched Predicted Next Sibling: {potential_structure}")
            elif potential_structure in self.predicted_subsections:
                matched_prediction_type = "subsection"
                self.debug.debug("match", f"Matched Predicted Subsection: {potential_structure}")
            else:
                parent_match_found = False
                for i in range(len(self.parent_sibling_prediction_stack) - 1, -1, -1):
                    parent_pred = self.parent_sibling_prediction_stack[i]
                    parent_match, _ = compare_structure(
                        potential_structure,
                        [list(t) for t in parent_pred] if parent_pred else None,
                    )
                    if parent_match:
                        matched_prediction_type = "parent_sibling"
                        matched_parent_level = i
                        parent_match_found = True
                        self.debug.debug(
                            "match",
                            f"Matched Predicted Parent Sibling (Stack Level {i}): {potential_structure}",
                        )
                        break
                if not parent_match_found:
                    if potential_structure in self.predicted_new_types:
                        matched_prediction_type = "new_section_type"
                        self.debug.debug(
                            "match",
                            f"MATCH FOUND: Potential header matches predicted new section type start for '{potential_handler.get_type_name()}': {potential_structure}",
                        )

        # --- Update state based on match ---
        if matched_prediction_type:
            self.debug.debug("state_update", f"Match type: {matched_prediction_type}")

            # Store prediction for OLD section's sibling BEFORE updating current_section_info
            old_sibling_prediction = self.predicted_next_sibling if matched_prediction_type == "subsection" else None

            # Finalize the previous section (if one exists)
            self._finalize_current_section(line_idx)

            # Update parent prediction stack based on match type BEFORE setting new current section
            if matched_prediction_type == "first":
                self.parent_sibling_prediction_stack = []
                self.first_section_found = True
                self.debug.debug("state_update", "Parent prediction stack cleared for first section.")
            elif matched_prediction_type == "subsection":
                # Push prediction for the level we are leaving
                self.parent_sibling_prediction_stack.append(old_sibling_prediction)
                self.debug.debug(
                    "state_update",
                    f"Pushed parent prediction: {old_sibling_prediction}. Parent stack size: {len(self.parent_sibling_prediction_stack)}",
                )
            elif matched_prediction_type == "parent_sibling":
                # Pop parent prediction stack based on matched level index 'i'
                levels_to_pop = len(self.parent_sibling_prediction_stack) - matched_parent_level
                self.debug.debug(
                    "state_update",
                    f"Popping {levels_to_pop} levels from parent prediction stack (matched level {matched_parent_level}).",
                )
                del self.parent_sibling_prediction_stack[matched_parent_level:]
                self.debug.debug(
                    "state_update",
                    f"Parent stack size after pop: {len(self.parent_sibling_prediction_stack)}",
                )
            elif matched_prediction_type == "new_section_type":
                # New top level type, clear parent predictions
                self.parent_sibling_prediction_stack = []
                self.first_section_found = True  # Ensure this stays true
                self.debug.debug(
                    "state_update",
                    "Parent prediction stack cleared for new section type.",
                )
            # Sibling match: No change to parent prediction stack

            # Set new current section info
            self.current_section_info = potential_info
            self.debug.debug(
                "state_update",
                f"Current section info set to: {self.current_section_info['S']}",
            )

            # Recalculate predictions based on the NEW current section
            self._calculate_predictions()

            # Reset text buffer for the NEW section
            self.current_section_text_lines = []
            self.debug.debug("state_update", "Reset text buffer for new section.")

        else:  # Parsed as header but no match found
            if self.current_section_info:
                current_struct_str = SectionOperationsHelper.get_structure_string(self.current_section_info["S"])
                self.debug.debug(
                    "match",
                    f"Potential header '{line_text}' ({potential_structure}) did not match any predictions. Treating as content for section '{current_struct_str}'.",
                )
                self.current_section_text_lines.append((page_num, original_line_text))
            else:
                self.debug.debug(
                    "match",
                    f"Potential header '{line_text}' ({potential_structure}) did not match policy or predictions and no section active. Discarding line.",
                )

    def finalize(self, total_lines: int):
        """Finalizes the last section."""
        self.debug.debug("state_logic", f"Finalizing processing. Last line index: {total_lines - 1}")
        self._finalize_current_section(total_lines)
        self.debug.debug(
            "state_logic",
            f"Finalization complete. Total processed sections: {len(self.processed_sections)}",
        )

    def get_results(self) -> List[OutputSection]:
        """Returns the list of processed sections."""
        self.debug.debug(
            "state_logic",
            f"get_results called. Returning {len(self.processed_sections)} processed sections.",
        )
        return self.processed_sections
