# src/pdf_parser/parsing.py

import os
import traceback
import pdfplumber
import langdetect
from enum import Enum
from typing import List, Tuple, Dict, Any

from .exceptions import PDFParsingError, PDFReadError, ParsingLogicError, ConfigError
from .config import load_config  # Use functions from config module
from .state import SectionProcessor  # Use the new state processor
from .debug import DebugManager  # Import debug manager
from .models import OutputSection, StructuredSection  # Import models


# --- Enums ---
class ParseMethod(str, Enum):
    PARSE_SECTIONS = "sections"
    PARSE_PAGES = "pages"


# --- Constants (Consider moving to config or a dedicated constants module) ---
MIN_DOC_LENGTH = 32
MIN_KEEP_LENGTH = 512
# N_LANG_CHUNKS = 16 # Not currently used in the simplified language check

langdetect.DetectorFactory.seed = 42  # Ensure reproducibility for langdetect


# --- PDF Extraction and Line Processing Utilities ---


def extract_data(pdf_path: str, debug_manager: DebugManager, prog_cb=None):
    """
    Extracts text lines (per page), images (per page), and metadata from PDF.
    """
    img_counter = 0
    # Stores list of lines per page: List[List[str]]
    pages_text_lines: List[List[str]] = []
    # Stores list of images per page: List[List[Dict]]
    pages_images: List[List[Dict]] = []
    metadata: Optional[Dict] = None

    debug_manager.debug("extract", f"Opening PDF: {pdf_path}")
    try:
        with pdfplumber.open(pdf_path) as pdf:
            metadata = pdf.metadata
            npages = len(pdf.pages)
            debug_manager.debug("extract", f"Processing {npages} pages...")
            for i, page in enumerate(pdf.pages):
                # Conceptually update line number for debug manager per page start
                debug_manager.process_line(
                    i * 1000, f"--- Page {i + 1} Start ---"
                )  # Use large line number jump per page?

                # if prog_cb and i % 10 == 0: prog_cb("reading_pdf", i / npages) # Progress callback if needed

                # Extract text, attempting to preserve layout within lines
                ptext = page.extract_text(
                    x_tolerance=2, y_tolerance=2, keep_blank_chars=True
                )
                page_lines = ptext.split("\n") if ptext else []
                pages_text_lines.append(page_lines)

                # Extract images
                pimages = []
                for im_counter_page, im in enumerate(page.images):
                    img_counter += 1
                    img_info = {
                        "name": f"Image_{img_counter}",  # Unique name
                        "page_index": i,  # 0-based page index
                        "page_image_index": im_counter_page,  # Index within the page
                        "x0": im.get("x0"),
                        "y0": im.get("y0"),
                        "x1": im.get("x1"),
                        "y1": im.get("y1"),
                        "height": im["height"],
                        "width": im["width"],
                    }
                    pimages.append(img_info)
                    debug_manager.debug(
                        "extract_img",
                        f"Found image {img_counter} on page {i} at index {im_counter_page}",
                    )
                pages_images.append(pimages)

                page.flush_cache()  # Release page resources

            debug_manager.debug(
                "extract", f"Finished extracting data from {npages} pages."
            )

    except pdfplumber.pdfminer.pdfdocument.PDFPasswordIncorrect as e:
        raise PDFReadError("PDF is password protected.", e) from e
    # Add more specific exceptions from pdfplumber/pdfminer if needed
    except Exception as e:
        raise PDFReadError(f"Error reading PDF: {e}", e) from e

    return {
        "pages_text_lines": pages_text_lines,
        "pages_images": pages_images,
        "metadata": metadata,
    }


def check_header(page_lines: List[str]) -> bool:
    """Placeholder for header detection logic on a single page."""
    # Example: Check if the first non-empty line is the same across pages
    # TODO: Implement actual header detection if needed
    return False


def check_footer(page_lines: List[str]) -> bool:
    """Placeholder for footer detection logic on a single page."""
    # Example: Check if the last non-empty line looks like a page number
    # TODO: Implement actual footer detection if needed
    # Simple check: Assume last line might be a footer if it's just digits
    stripped_lines = [line.strip() for line in page_lines if line.strip()]
    # if stripped_lines and stripped_lines[-1].isdigit():
    #      return True # Simple assumption
    return False  # Disable footer removal by default for now


def get_lines(
    pages_data: List[List[str]], debug_manager: DebugManager
) -> List[Tuple[int, str]]:
    """
    Converts extracted page data into a list of (page_index, line_text) tuples,
    applying basic header/footer removal if detected.
    """
    debug_manager.debug(
        "line_proc", f"Starting get_lines for {len(pages_data)} pages..."
    )
    all_lines: List[Tuple[int, str]] = []

    for i, page_lines in enumerate(pages_data):
        lines_to_process = list(page_lines)
        original_count = len(lines_to_process)

        # Apply header/footer logic per page
        # Note: More robust logic might compare first/last lines across pages
        remove_header = check_header(lines_to_process)
        remove_footer = check_footer(lines_to_process)

        if remove_header and lines_to_process:
            lines_to_process = lines_to_process[1:]
            debug_manager.debug(
                "line_proc", f"Page {i}: Removed potential header line."
            )
        if remove_footer and lines_to_process:
            lines_to_process = lines_to_process[:-1]
            debug_manager.debug(
                "line_proc", f"Page {i}: Removed potential footer line."
            )

        # Add remaining lines with page index
        for line in lines_to_process:
            all_lines.append((i, line))  # Store original line text

    debug_manager.debug(
        "line_proc", f"get_lines finished. Total lines processed: {len(all_lines)}"
    )
    return all_lines


def perform_language_check(text: str, debug_manager: DebugManager) -> bool:
    """Checks if text is likely English. Returns True if OK, False otherwise."""
    # Use constants defined at module level or loaded from config
    if len(text) < MIN_KEEP_LENGTH:
        debug_manager.debug(
            "lang_check",
            f"Skipping language check for short text (len {len(text)} < {MIN_KEEP_LENGTH})",
        )
        return True

    sample = text[:50000]  # Limit sample size for performance/robustness
    try:
        lang = langdetect.detect(sample)
        debug_manager.debug("lang_check", f"Detected language: {lang}")
        if lang != "en":
            # Add more sophisticated checks (e.g., language probabilities, proportion) if needed
            debug_manager.debug(
                "lang_check", f"Language '{lang}' is not 'en'. Filtering section/page."
            )
            return False
    except langdetect.lang_detect_exception.LangDetectException:
        # langdetect can fail on short/ambiguous text
        debug_manager.debug(
            "lang_check",
            "Language detection failed (likely ambiguous text). Assuming OK.",
        )
        pass  # Assume English if detection fails
    except Exception as e:
        debug_manager.debug("lang_check", f"ERROR during language detection: {e}")
        pass  # Assume English on other errors

    return True


# --- Main Parsing Function ---


def to_sections(
    pdf_path: str,
    config: Dict,
    debug_manager: DebugManager,
    parse_method: ParseMethod = ParseMethod.PARSE_SECTIONS,
    prog_cb=None,
) -> Dict[str, Any]:
    """
    Main function to parse PDF into sections based on configuration and method.
    Uses the SectionProcessor state machine for section detection.
    """
    debug_manager.debug(
        "main",
        f"Starting to_sections for: {pdf_path} using method: {parse_method.value}",
    )
    final_sections: List[OutputSection] = []
    extracted_data: Dict[str, Any] = {}  # Ensure defined scope

    try:
        # --- Step 1: Extract Data from PDF ---
        extracted_data = extract_data(pdf_path, debug_manager, prog_cb)
        # if prog_cb: prog_cb("parsing_sections", 0.1) # Update progress estimation

        # --- Step 2: Process based on Method ---
        if parse_method == ParseMethod.PARSE_PAGES:
            debug_manager.debug("main", "Parsing by pages...")
            num_pages = len(extracted_data["pages_text_lines"])
            for i, page_content in enumerate(extracted_data["pages_text_lines"]):
                # Update debug manager line context conceptually per page
                debug_manager.process_line(
                    i * 1000, f"--- Processing Page {i + 1} Content ---"
                )
                page_text = "\n".join(page_content)

                # Perform language check per page
                if perform_language_check(page_text, debug_manager):
                    # Find images for this page
                    page_images = (
                        extracted_data["pages_images"][i]
                        if i < len(extracted_data["pages_images"])
                        else []
                    )

                    # Create dummy structure for page-based sections
                    page_structure: StructuredSection = [("Page", i + 1)]

                    output_sec: OutputSection = {
                        "section_title": f"Page {i + 1}",
                        "section_text": page_text.strip(),  # Strip final whitespace
                        "images": page_images,
                        "page_number": i,  # 0-based page index
                        "structure": page_structure,
                        "type": "Page",
                    }
                    final_sections.append(output_sec)
                else:
                    debug_manager.debug(
                        "main", f"Skipping Page {i + 1} due to language check."
                    )

            debug_manager.debug(
                "main", f"Generated {len(final_sections)} sections from pages."
            )

        elif parse_method == ParseMethod.PARSE_SECTIONS:
            debug_manager.debug(
                "main", "Parsing by detected sections using predictive state machine..."
            )

            # Initialize handlers (now done inside SectionProcessor)
            # Need to import handlers here if SectionProcessor expects instances
            from .handlers import (
                NumericHandler,
                AlphaNumericHandler,
                SectionHandler,
            )

            handlers_map: Dict[str, SectionHandler] = {}
            available_handler_names = config.get(
                "section_handlers", []
            )  # Names from YAML
            debug_manager.debug(
                "main", f"Configured handlers: {available_handler_names}"
            )

            if "Numeric" in available_handler_names:
                handlers_map["Numeric"] = NumericHandler(type_name="Numeric")

            if "AlphaNumeric" in available_handler_names:
                handlers_map["AlphaNumeric"] = AlphaNumericHandler(
                    type_name="AlphaNumeric"
                )

            if "Annex" in available_handler_names:
                # Directly initialize AlphaNumericHandler with Annex details
                handlers_map["Annex"] = AlphaNumericHandler(
                    type_name="Annex",
                    prefix="Annex",
                    allow_trailing_dot=True,  # Allow "Annex A."
                )

            if "Appendix" in available_handler_names:
                # Directly initialize AlphaNumericHandler (or NumericHandler) with Appendix details
                # Example: Using AlphaNumeric base, allowing trailing dot
                handlers_map["Appendix"] = AlphaNumericHandler(
                    type_name="Appendix",
                    prefix="Appendix",
                    allow_trailing_dot=True,  # Allow "Appendix A."
                )
                # Example if Appendix used Numeric base:
                # handlers_map["Appendix"] = NumericHandler(
                #    type_name="Appendix",
                #    prefix="Appendix",
                #    allow_trailing_dot=True # Allow "Appendix 1."
                # )

            # Filter handlers_map to only include those specified in config
            active_handlers_map = {
                name: handler
                for name, handler in handlers_map.items()
                if name in available_handler_names
            }

            if not active_handlers_map:
                if available_handler_names:
                    raise ConfigError(
                        f"Specified handlers in config ({available_handler_names}) did not match known implementation types."
                    )
                else:
                    raise ConfigError(
                        "No section handlers specified or enabled in configuration."
                    )

            debug_manager.debug(
                "main",
                f"Initialized active handlers: {list(active_handlers_map.keys())}",
            )

            lines = get_lines(extracted_data["pages_text_lines"], debug_manager)
            # if prog_cb: prog_cb("parsing_sections", 0.3)

            # Initialize and run the state processor
            state_processor = SectionProcessor(config, handlers_map, debug_manager)
            for i, line_info in enumerate(lines):
                state_processor.process_line(line_info, i)
                # if prog_cb and i % 100 == 0: prog_cb("parsing_sections", 0.3 + 0.6 * (i / len(lines)))

            state_processor.finalize(len(lines))
            raw_sections = state_processor.get_results()  # Get raw sections

            # Assign images to sections based on page numbers
            # This requires linking section line ranges back to page numbers
            # TODO: Implement image assignment logic here if needed.
            # For now, images remain empty in the output sections.
            final_sections = raw_sections  # Use raw sections for now

            debug_manager.debug(
                "main",
                f"Generated {len(final_sections)} sections from predictive detection.",
            )

        else:
            # Should not happen if using Enum, but safeguard
            raise ParsingLogicError(
                f"Unknown parsing method specified: {parse_method.value}"
            )

        # if prog_cb: prog_cb("parsing_sections", 1.0)

    except PDFParsingError as e:
        # Log known parsing errors
        debug_manager.debug(
            "error", f"Caught PDFParsingError: Code='{e.code}', Message='{e.message}'"
        )
        raise  # Re-raise known errors
    except Exception as e:
        # Catch unexpected errors during processing
        trace = "".join(traceback.format_tb(e.__traceback__))
        msg = f"Unexpected error during parsing logic: {str(e)}\n\n{trace}"
        debug_manager.debug("error", msg)
        # Wrap unexpected errors
        raise ParsingLogicError(msg) from e

    # --- Final check for sufficient content ---
    total_text_len = sum(len(s.get("section_text", "")) for s in final_sections)
    if not final_sections or total_text_len < MIN_DOC_LENGTH:
        raise PDFParsingError(
            "too_little_text",
            f"Parsing resulted in too little text content (Total chars: {total_text_len}). Final section count: {len(final_sections)}",
        )

    debug_manager.debug("main", "to_sections completed successfully.")
    return {
        "filename": os.path.basename(pdf_path),
        "metadata": extracted_data.get("metadata"),  # Get metadata if extracted
        "sections": final_sections,
    }
