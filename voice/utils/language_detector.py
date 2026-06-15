"""
Language Detection
Uses lingua-language-detector for better accuracy on short texts
Falls back to langdetect if lingua is not available
"""

import threading
from typing import Optional

from .logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level lingua detector singleton
# Lingua's LanguageDetector is expensive to build (~100 ms) so we construct
# it once and reuse it across all calls.
# A threading.Lock guards against double-initialization when multiple threads
# call _get_lingua_detector() simultaneously.
# ---------------------------------------------------------------------------

_lingua_detector = None
_lingua_available: Optional[bool] = None  # None = not yet probed
_lingua_lock = threading.Lock()


def _get_lingua_detector():
    """Return the cached lingua detector, building it on first access.

    Returns None if lingua is not installed.
    """
    global _lingua_detector, _lingua_available

    # Fast path: already probed – return cached outcome without acquiring lock
    if _lingua_available is False:
        return None
    if _lingua_detector is not None:
        return _lingua_detector

    with _lingua_lock:
        # Re-check after acquiring lock (double-checked locking pattern)
        if _lingua_available is False:
            return None
        if _lingua_detector is not None:
            return _lingua_detector

        try:
            from lingua import LanguageDetectorBuilder

            logger.debug("[language_detector] Building lingua detector …")
            _lingua_detector = (
                LanguageDetectorBuilder.from_all_languages()
                .with_minimum_relative_distance(0.9)
                .build()
            )
            _lingua_available = True
            logger.info("[language_detector] lingua detector ready.")
            return _lingua_detector

        except ImportError:
            _lingua_available = False
            logger.info(
                "[language_detector] lingua not installed; falling back to langdetect. "
                "For better accuracy on short texts install: pip install lingua-language-detector"
            )
            return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_language(text: str, confidence_threshold: float = 0.7) -> str:
    """Detect the language of *text* and return its ISO 639-1 code.

    Detection strategy
    ------------------
    1. Try **lingua** (more accurate on short/ambiguous text).
    2. Fall back to **langdetect** if lingua is unavailable.
    3. Return ``"en"`` if both fail or text is empty.

    Parameters
    ----------
    text:
        The text whose language should be detected.
    confidence_threshold:
        Minimum confidence required when using lingua's confidence-based
        API.  Below this threshold the result from the primary detector is
        still returned (lingua's ``detect_language_of`` already applies an
        internal minimum relative distance), but callers can use this value
        to decide whether to trust the result.  Currently this parameter
        influences only the logging verbosity.

    Returns
    -------
    str
        ISO 639-1 two-letter language code (lower-case), e.g. ``"en"``,
        ``"fr"``, ``"zh"``.  Returns ``"en"`` as a safe fallback.
    """
    if not text or not text.strip():
        logger.debug("[language_detector] Empty text – returning fallback 'en'.")
        return "en"

    # --- Attempt 1: lingua ---
    lingua_detector = _get_lingua_detector()
    if lingua_detector is not None:
        try:
            language = lingua_detector.detect_language_of(text)
            if language is not None:
                # lingua's iso_code_639_1 attribute is an enum; .name gives
                # the two-letter code in upper-case (e.g. "EN").
                code = language.iso_code_639_1.name.lower()
                logger.debug(
                    f"[language_detector] lingua detected '{code}' "
                    f"for text: {text[:60]!r}"
                )
                return code
            logger.debug(
                "[language_detector] lingua returned None – trying langdetect."
            )
        except Exception as exc:
            logger.warning(
                f"[language_detector] lingua error ({exc}) – trying langdetect."
            )

    # --- Attempt 2: langdetect ---
    try:
        from langdetect import detect, LangDetectException

        code = detect(text)
        logger.debug(
            f"[language_detector] langdetect detected '{code}' "
            f"for text: {text[:60]!r}"
        )
        return code
    except ImportError:
        logger.warning(
            "[language_detector] langdetect not installed. "
            "Install with: pip install langdetect"
        )
    except Exception as exc:
        logger.warning(
            f"[language_detector] langdetect error ({exc}) – returning fallback 'en'."
        )

    # --- Final fallback ---
    logger.debug("[language_detector] All detectors failed – returning 'en'.")
    return "en"
