"""Text sanitization for TTS — strip markdown, emoji, and special characters."""

import re


# Unicode ranges for emojis and symbols
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0010FFFF"  # Emoticons, symbols, supplemental
    "☀-➿"           # Misc symbols, dingbats
    "⌀-⏿"           # Misc technical (including ⏰ etc.)
    "─-╿"           # Box drawing
    "⬀-⯿"           # Misc symbols and arrows
    "✀-➿"           # Dingbats
    "︀-️"           # Variation selectors
    "‍"                  # Zero-width joiner
    "]"
)

# Markdown patterns to strip
_MARKDOWN_PATTERNS = [
    re.compile(r"```[\s\S]*?```"),           # Code blocks
    re.compile(r"`[^`]+`"),                  # Inline code
    re.compile(r"!?\[([^\]]*)\]\([^)]*\)"),  # Links and images -> keep text
    re.compile(r"^#{1,6}\s+", re.MULTILINE), # Headers
    re.compile(r"^[-*+]\s+", re.MULTILINE),  # Unordered list items
    re.compile(r"^\d+[.)]\s+", re.MULTILINE), # Ordered list items
    re.compile(r"^>\s+", re.MULTILINE),       # Blockquotes
    re.compile(r"(\*{1,3}|_{1,3})"),         # Bold/italic markers
    re.compile(r"~~(.+?)~~"),                # Strikethrough
    re.compile(r"\|"),                       # Table separators
]

# Special characters that cause TTS artifacts
_SPECIAL_CHARS_PATTERN = re.compile(r"[~～^_`#*|\\]")

# Multiple whitespace
_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")


def sanitize_for_tts(text: str) -> str:
    """Strip markdown, emoji, and special characters from LLM output.

    Designed for TTS consumption — keeps Chinese/English text and
    standard punctuation, removes anything that would cause audio artifacts.
    """
    result = text

    # 1. Remove markdown code blocks first (they may contain special chars)
    for pattern in _MARKDOWN_PATTERNS:
        result = pattern.sub("", result)

    # 2. Handle markdown links: keep only the display text
    result = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", result)

    # 3. Remove emojis
    result = _EMOJI_PATTERN.sub("", result)

    # 4. Remove special characters
    result = _SPECIAL_CHARS_PATTERN.sub("", result)

    # 5. Collapse whitespace
    result = _MULTI_NEWLINE.sub("\n\n", result)
    result = _MULTI_SPACE.sub(" ", result)

    return result.strip()
