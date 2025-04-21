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
    def __init__(
        self,
        config: Dict,
        handlers: Dict[str, "SectionHandler"],
        debug_manager: DebugManager,
    ):  # Removed handler_min_levels if not using that approach
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

        self.debug.debug(
            "state_init",
            f"SectionProcessor initialized. Handlers: {list(self.handlers.keys())}",
        )
        # Initial prediction based on policy (and potentially processed rules)
        self._calculate_predictions()

    # --- CONFIG RULES PROCESSING ---

    def _process_config_rules(self):
        """Post-process config rules, parsing required_start_structure strings."""
        self.debug.debug(
            "config_proc",
            "Post-processing config rules for required_start_structure strings...",
        )
        handler_check_order = list(
            self.config.get(
                "handler_check_order", ["Annex", "Appendix", "Numeric", "AlphaNumeric"]
            )
        )
        if not self.rules:
            self.debug.debug(
                "config_proc", "No 'valid_next_section_rules' found in config."
            )
            return
        for type_name, rule_list in self.rules.items():
            if not isinstance(rule_list, list):
                continue
            for rule_idx, rule in enumerate(rule_list):
                self._process_single_rule(
                    type_name, rule, rule_idx, handler_check_order
                )

    def _process_single_rule(self, type_name, rule, rule_idx, handler_check_order):
        """Process a single rule for required_start_structure parsing."""
        op = rule.get("operation")
        req_struct_val = rule.get("required_start_structure")
        if op == "new_section_type" and req_struct_val:
            structure_strings_to_parse = self._normalize_structure_strings(
                req_struct_val
            )
            if structure_strings_to_parse is None:
                self.debug.debug(
                    "config_proc",
                    f"  WARNING: Invalid format for required_start_structure in rule #{rule_idx + 1} for '{type_name}'. Expected string or list of strings. Value: {req_struct_val}",
                )
                return
            parsed_struct_list = []
            for structure_string in structure_strings_to_parse:
                parsed_struct = parse_identifier_string(
                    structure_string, self.handlers, handler_check_order
                )
                if parsed_struct:
                    parsed_struct_list.append(parsed_struct)
                    self.debug.debug(
                        "config_proc", f"    Successfully parsed to: {parsed_struct}"
                    )
                else:
                    self.debug.debug(
                        "config_proc",
                        f"    WARNING: Failed to parse required_start_structure string '{structure_string}'. It will be ignored for this rule.",
                    )
            rule["required_start_structure"] = parsed_struct_list

    def _normalize_structure_strings(self, req_struct_val):
        """Normalize required_start_structure to a list of strings, or None if invalid."""
        if isinstance(req_struct_val, str):
            return [req_struct_val]
        elif isinstance(req_struct_val, list) and all(
            isinstance(s, str) for s in req_struct_val
        ):
            return req_struct_val
        elif isinstance(req_struct_val, list):
            self.debug.debug(
                "config_proc",
                f"  Found existing list for required_start_structure. Assuming pre-parsed or will be handled later.",
            )
            return None
        return None

    # --- PREDICTION CALCULATION ---

    def _calculate_predictions(self):
        """Calculates predictions, using pre-parsed required_start_structure from rules."""
        current_section_info = self.current_section_info
        current_struct_str = (
            SectionOperationsHelper.get_structure_string(current_section_info["S"])
            if current_section_info
            else "None"
        )
        self.debug.debug(
            "prediction",
            f"Calculating predictions based on current: {current_struct_str}",
        )

        self._reset_predictions()
        if not current_section_info:
            self._predict_first_section()
            return

        current_structure = current_section_info["S"]
        current_type = current_section_info["T"]

        self._predict_next_sibling(current_structure)
        self._predict_subsections(current_structure, current_type)
        self._predict_new_types(current_type)

        self.debug.debug(
            "prediction", f"--- End Predictions for {current_struct_str} ---"
        )

    def _reset_predictions(self):
        self.predicted_next_sibling = None
        self.predicted_subsections = []
        self.predicted_new_types = []

    def _predict_first_section(self):
        if not self.policy.get("accept_any", False):
            must_struct_config = self.policy.get("must_start_with_structure")
            if must_struct_config and isinstance(must_struct_config, list):
                try:
                    self.predicted_next_sibling = [
                        tuple(item) for item in must_struct_config
                    ]
                    self.debug.debug(
                        "prediction",
                        f"Predicting REQUIRED first section based on policy: {self.predicted_next_sibling}",
                    )
                except Exception as e:
                    self.debug.debug(
                        "prediction",
                        f"Error creating first section prediction from policy: {e}",
                    )
                    self.predicted_next_sibling = None
        self.debug.debug(
            "prediction",
            f"Initial state predictions: Required First={self.predicted_next_sibling}, Subsections=[], New Types=[]",
        )
        self.debug.debug("prediction", f"--- End Predictions (Initial State) ---")

    def _predict_next_sibling(self, current_structure):
        if current_structure:
            last_type, last_val = current_structure[-1]
            next_val = SectionOperationsHelper.increment_value(last_val)
            if next_val is not None:
                self.predicted_next_sibling = current_structure[:-1] + [
                    (last_type, next_val)
                ]
        self.debug.debug(
            "prediction",
            f"Predicted next sibling: {SectionOperationsHelper.get_structure_string(self.predicted_next_sibling) if self.predicted_next_sibling else 'None'}",
        )

    def _predict_subsections(self, current_structure, current_type):
        self.predicted_subsections = []
        numeric_subsection_struct = current_structure + [("N", 1)]
        self.predicted_subsections.append(numeric_subsection_struct)
        if current_structure and current_structure[-1][0] != "A":
            alpha_subsection_struct = current_structure + [("A", "A")]
            self.predicted_subsections.append(alpha_subsection_struct)
        self.debug.debug(
            "prediction",
            f"  Added conventional subsection predictions: {[SectionOperationsHelper.get_structure_string(s) for s in self.predicted_subsections]}",
        )
        allowed_rules = self.rules.get(current_type, [])
        sub_type_rules = [
            rule
            for rule in allowed_rules
            if rule.get("operation") == "new_subsection_type"
        ]
        for rule_idx, rule in enumerate(sub_type_rules):
            self._process_subsection_rule(rule, current_structure, rule_idx)
        self.debug.debug(
            "prediction",
            f"Final Predicted subsections: {[SectionOperationsHelper.get_structure_string(s) for s in self.predicted_subsections]}",
        )

    def _process_subsection_rule(self, rule, current_structure, rule_idx):
        params = rule.get("params")
        target_type = rule.get("target_type")
        if not params or not target_type:
            self.debug.debug(
                "prediction_detail", "    Rule missing params or target_type, skipping."
            )
            return
        core_structure = list(current_structure)
        if core_structure and core_structure[0][0] == "P":
            core_structure = core_structure[1:]
        allowed_types = params.get("allowed_types")
        start_value = params.get("start_with")
        if not allowed_types or start_value is None:
            self.debug.debug(
                "prediction_detail",
                f"    Rule missing allowed_types or start_with in params, skipping: {params}",
            )
            return
        new_level_type_config = allowed_types[0]
        new_comp_type = (
            "N"
            if new_level_type_config == "Numeric"
            else "A"
            if new_level_type_config == "Alpha"
            else None
        )
        if new_comp_type:
            new_structure_no_prefix = core_structure + [(new_comp_type, start_value)]
            if new_structure_no_prefix not in self.predicted_subsections:
                self.predicted_subsections.append(new_structure_no_prefix)
                self.debug.debug(
                    "prediction_detail",
                    f"    **ADDED** subsection prediction (from rule, prefix dropped): {new_structure_no_prefix}",
                )

    def _predict_new_types(self, current_type):
        self.predicted_new_types = []
        allowed_rules = self.rules.get(current_type, [])
        transition_rules = [
            rule
            for rule in allowed_rules
            if rule.get("operation") == "new_section_type"
        ]
        processed_targets = set()
        for rule in transition_rules:
            target_type_name = rule.get("target_type")
            if not target_type_name:
                continue
            required_starts = rule.get("required_start_structure")
            if isinstance(required_starts, list):
                for struct in required_starts:
                    if struct not in self.predicted_new_types:
                        self.predicted_new_types.append(struct)
                processed_targets.add(target_type_name)
            elif target_type_name not in processed_targets:
                target_handler = self.handlers.get(target_type_name)
                if target_handler:
                    default_start = target_handler.get_default_start_structure()
                    if default_start and default_start not in self.predicted_new_types:
                        self.predicted_new_types.append(default_start)
                processed_targets.add(target_type_name)
        self.debug.debug(
            "prediction",
            f"Final Predicted new types: {[SectionOperationsHelper.get_structure_string(s) for s in self.predicted_new_types]}",
        )

    # --- LANGUAGE CHECK ---

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
                self.debug.debug(
                    "lang_check", f"Language '{lang}' is not 'en'. Filtering section."
                )
                return False
        except langdetect.lang_detect_exception.LangDetectException:
            self.debug.debug("lang_check", "Language detection failed. Assuming OK.")
            pass
        except Exception as e:
            self.debug.debug("lang_check", f"ERROR during language detection: {e}")
            pass
        return True

    # --- FINALIZATION ---

    def _finalize_current_section(self, end_line_idx: int):
        """Adds the completed current section to the results list."""
        if not self.current_section_info:
            self.debug.debug(
                "finalize", "Called with no current section info. Nothing to finalize."
            )
            return

        current_struct_str = SectionOperationsHelper.get_structure_string(
            self.current_section_info["S"]
        )
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
        last_title = (
            self.processed_sections[-1]["section_title"]
            if self.processed_sections
            else "N/A"
        )
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

    # --- LINE PROCESSING ---

    def process_line(self, line_info: Tuple[int, str], line_idx: int):
        """
        Processes a single line, checking against predictions, updating state,
        and collecting text content using the minimal state approach.
        """
        self.line_counter = line_idx
        self.debug.process_line(line_idx, line_info[1])
        page_num, original_line_text = line_info
        line_text = original_line_text.strip()
        self.debug.debug(
            "line_proc", f"Processing Line {line_idx + 1}: '{line_text[:80]}...'"
        )
        if self._should_skip_line(line_text):
            self._handle_skipped_line(page_num, original_line_text)
            return
        potential_structure, potential_handler = self._try_parse_header(line_text)
        if potential_structure is None:
            self._handle_non_header_line(page_num, original_line_text)
            return
        self._handle_potential_header(
            potential_structure,
            potential_handler,
            line_text,
            line_idx,
            page_num,
            original_line_text,
        )

    def _should_skip_line(self, line_text: str) -> bool:
        max_dots = self.constants.get("MAX_DOTS", 16)
        return not line_text or line_text.count(".") >= max_dots

    def _handle_skipped_line(self, page_num, original_line_text):
        if self.current_section_info:
            self.current_section_text_lines.append((page_num, original_line_text))
            current_struct_str = SectionOperationsHelper.get_structure_string(
                self.current_section_info["S"]
            )
            self.debug.debug(
                "line_proc",
                f"Line added as content (blank/dots) to section '{current_struct_str}'",
            )

    def _try_parse_header(self, line_text: str):
        handler_check_order = ["Annex", "Appendix", "Numeric", "AlphaNumeric"]
        handlers_to_check = [
            self.handlers[name] for name in handler_check_order if name in self.handlers
        ]
        for handler in handlers_to_check + [
            self.handlers[name]
            for name in self.handlers
            if name not in handler_check_order
        ]:
            if handler.get_regex().match(line_text):
                parsed = handler.parse(line_text)
                if parsed:
                    return parsed, handler
        return None, None

    def _handle_non_header_line(self, page_num, original_line_text):
        if self.current_section_info:
            self.current_section_text_lines.append((page_num, original_line_text))
            current_struct_str = SectionOperationsHelper.get_structure_string(
                self.current_section_info["S"]
            )
            self.debug.debug(
                "line_proc", f"Line added as content to section '{current_struct_str}'"
            )
        else:
            self.debug.debug(
                "line_proc",
                "Line is not a header and no current section active, discarding.",
            )

    def _handle_potential_header(
        self,
        potential_structure,
        potential_handler,
        line_text,
        line_idx,
        page_num,
        original_line_text,
    ):
        matched_prediction_type, matched_parent_level = self._determine_match_type(
            potential_structure, potential_handler
        )
        potential_info: StackElement = {
            "S": potential_structure,
            "T": potential_handler.get_type_name(),
            "text": line_text,
            "line_idx": line_idx,
        }
        if matched_prediction_type:
            self._update_state_on_match(
                matched_prediction_type, matched_parent_level, line_idx, potential_info
            )
        else:
            self._handle_unmatched_header(
                page_num, original_line_text, line_text, potential_structure
            )

    def _determine_match_type(self, potential_structure, potential_handler):
        matched_prediction_type = None
        matched_parent_level = -1
        if not self.first_section_found:
            policy_struct_prediction = self.predicted_next_sibling
            policy_struct_config_format = (
                [list(t) for t in policy_struct_prediction]
                if policy_struct_prediction
                else None
            )
            match_status, _ = compare_structure(
                potential_structure, policy_struct_config_format
            )
            if match_status or self.policy.get("accept_any", False):
                matched_prediction_type = "first"
        else:
            sibling_match, _ = compare_structure(
                potential_structure,
                [list(t) for t in self.predicted_next_sibling]
                if self.predicted_next_sibling
                else None,
            )
            if sibling_match:
                matched_prediction_type = "sibling"
            elif potential_structure in self.predicted_subsections:
                matched_prediction_type = "subsection"
            else:
                for i in range(len(self.parent_sibling_prediction_stack) - 1, -1, -1):
                    parent_pred = self.parent_sibling_prediction_stack[i]
                    parent_match, _ = compare_structure(
                        potential_structure,
                        [list(t) for t in parent_pred] if parent_pred else None,
                    )
                    if parent_match:
                        matched_prediction_type = "parent_sibling"
                        matched_parent_level = i
                        break
                if (
                    not matched_prediction_type
                    and potential_structure in self.predicted_new_types
                ):
                    matched_prediction_type = "new_section_type"
        return matched_prediction_type, matched_parent_level

    def _update_state_on_match(
        self, matched_prediction_type, matched_parent_level, line_idx, potential_info
    ):
        old_sibling_prediction = (
            self.predicted_next_sibling
            if matched_prediction_type == "subsection"
            else None
        )
        self._finalize_current_section(line_idx)
        if matched_prediction_type == "first":
            self.parent_sibling_prediction_stack = []
            self.first_section_found = True
        elif matched_prediction_type == "subsection":
            self.parent_sibling_prediction_stack.append(old_sibling_prediction)
        elif matched_prediction_type == "parent_sibling":
            del self.parent_sibling_prediction_stack[matched_parent_level:]
        elif matched_prediction_type == "new_section_type":
            self.parent_sibling_prediction_stack = []
            self.first_section_found = True
        self.current_section_info = potential_info
        self._calculate_predictions()
        self.current_section_text_lines = []

    def _handle_unmatched_header(
        self, page_num, original_line_text, line_text, potential_structure
    ):
        if self.current_section_info:
            current_struct_str = SectionOperationsHelper.get_structure_string(
                self.current_section_info["S"]
            )
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

    # --- FINALIZATION AND RESULTS ---

    def finalize(self, total_lines: int):
        """Finalizes the last section."""
        self.debug.debug(
            "state_logic", f"Finalizing processing. Last line index: {total_lines - 1}"
        )
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
