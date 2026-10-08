def int_to_gematria(num: int) -> str:
    """
    Converts a positive integer (up to 999) to its Hebrew Gematria representation.
    Correctly handles special combinations like 15 (ט"ו) and 16 (ט"ז) to avoid Yud-Heh/Yud-Vav.
    Formats multi-letter strings with double quotes before the last character.
    """
    if num <= 0 or num >= 1000:
        raise ValueError("Gematria conversion is only supported for integers between 1 and 999.")

    units = ["", "א", "ב", "ג", "ד", "ה", "ו", "ז", "ח", "ט"]
    tens = ["", "י", "כ", "ל", "מ", "נ", "ס", "ע", "פ", "צ"]
    hundreds = ["", "ק", "ר", "ש", "ת", "תק", "תר", "תש", "תת", "תתק"]

    h = num // 100
    t = (num % 100) // 10
    u = num % 10

    # Special cases for 15 and 16
    if t == 1 and u == 5:
        res = hundreds[h] + "טו"
    elif t == 1 and u == 6:
        res = hundreds[h] + "טז"
    else:
        res = hundreds[h] + tens[t] + units[u]

    # Format with quotes
    if len(res) > 1:
        return res[:-1] + '"' + res[-1]
    elif len(res) == 1:
        return res + "'"
    return ""

import re
from typing import Optional

HEBREW_GEMATRIA_VALUES = {
    'א': 1, 'ב': 2, 'ג': 3, 'ד': 4, 'ה': 5, 'ו': 6, 'ז': 7, 'ח': 8, 'ט': 9,
    'י': 10, 'כ': 20, 'ך': 20, 'ל': 30, 'מ': 40, 'ם': 40, 'נ': 50, 'ן': 50,
    'ס': 60, 'ע': 70, 'פ': 80, 'ף': 80, 'צ': 90, 'ץ': 90,
    'ק': 100, 'ר': 200, 'ש': 300, 'ת': 400
}

def gematria_to_int(text: str) -> Optional[int]:
    """
    Converts a Hebrew gematria string or digit string (e.g. 'יא', 'י"א', 'י״א', '11') to an integer.
    Returns None if the string cannot be converted.
    """
    if not text:
        return None
    clean = re.sub(r'["״\'׳’`\s]', '', str(text))
    if clean.isdigit():
        val = int(clean)
        return val if val > 0 else None
    if not clean:
        return None
    val = 0
    for char in clean:
        if char in HEBREW_GEMATRIA_VALUES:
            val += HEBREW_GEMATRIA_VALUES[char]
        else:
            return None
    return val if val > 0 else None
