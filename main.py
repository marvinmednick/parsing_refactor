import argparse
import os
import sys
from datetime import datetime

# Add src to path if running directly for development
# In a proper installation, this shouldn't be necessary
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from pdf_parser.parsing import to_sections, ParseMethod
from pdf_parser.exceptions import PDFParsingError
from pdf_parser.config import load_config, load_debug_config
from pdf_parser.debug import (
    DebugManager,
    CategoryDebugConfig,
)  # Import necessary classes


def main():
    parser = argparse.ArgumentParser(description="Parse PDF documents into sections.")
    parser.add_argument("pdf_path", help="Path to the input PDF file.")
    parser.add_argument(
        "-c",
        "--config",
        default="config.yaml",
        help="Path to the main configuration YAML file.",
    )
    parser.add_argument(
        "-d",
        "--debug-config",
        default=None,
        help="Path to the debug configuration YAML file (optional).",
    )
    parser.add_argument(
        "-m",
        "--method",
        choices=[e.value for e in ParseMethod],
        default=ParseMethod.PARSE_SECTIONS,
        help="Parsing method (sections or pages).",
    )
    parser.add_argument(
        "-o", "--output", default=None, help="Path to save the output JSON (optional)."
    )

    args = parser.parse_args()

    print(f"[{datetime.now()}] Starting processing for: {args.pdf_path}")

    try:
        # --- Load Configs ---
        print(f"Loading main config from: {args.config}")
        config = load_config(args.config)

        debug_manager = DebugManager()  # Create DebugManager instance
        if args.debug_config:
            print(f"Loading debug config from: {args.debug_config}")
            debug_categories = load_debug_config(args.debug_config)
            debug_manager.configure(debug_categories)
            print(f"Debug categories loaded: {list(debug_categories.keys())}")
        else:
            print("No debug config specified, debugging disabled.")

        # --- Run Parsing ---
        parsing_method = ParseMethod(args.method)
        result = to_sections(
            pdf_path=args.pdf_path,
            config=config,
            debug_manager=debug_manager,  # Pass the debug manager
            parse_method=parsing_method,
            prog_cb=None,  # Progress callback not implemented in this version
        )

        print(
            f"[{datetime.now()}] Successfully parsed {len(result.get('sections', []))} sections."
        )

        # --- Output Result ---
        if args.output:
            import json  # Use standard json for output

            print(f"Saving output to: {args.output}")
            try:
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"Error saving output file: {e}")
        else:
            # Print summary or part of the result if no output file
            print("\n--- Result Summary ---")
            print(f"Filename: {result.get('filename')}")
            print(f"Metadata: {result.get('metadata')}")
            print(f"Sections Found: {len(result.get('sections', []))}")
            if result.get("sections"):
                print("\nFirst 3 Section Titles:")
                for i, sec in enumerate(result["sections"][:3]):
                    print(f"  {i + 1}. {sec.get('section_title', 'N/A')}")

    except PDFParsingError as e:
        print(
            f"\n[{datetime.now()}] PDF Parsing Error ({e.code}): {e.message}",
            file=sys.stderr,
        )
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"\n[{datetime.now()}] File Not Found Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(
            f"\n[{datetime.now()}] An unexpected error occurred: {e}", file=sys.stderr
        )
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
