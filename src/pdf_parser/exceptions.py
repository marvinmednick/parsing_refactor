# Define custom exceptions for the parser


class PDFParsingError(Exception):
    """Base exception for errors during PDF parsing."""

    def __init__(
        self, code: str, message: str = "An unspecified parsing error occurred."
    ):
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")


class ConfigError(PDFParsingError):
    """Errors related to configuration loading or validation."""

    def __init__(self, message: str):
        super().__init__("config_error", message)


class ParsingLogicError(PDFParsingError):
    """Errors related to the section detection or processing logic."""

    def __init__(self, message: str):
        super().__init__("parsing_logic_error", message)


class PDFReadError(PDFParsingError):
    """Errors related to reading or interpreting the PDF file itself."""

    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__("pdf_read_error", message)
        self.original_exception = original_exception
