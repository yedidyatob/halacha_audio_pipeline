def int_to_gematria(num: int) -> str:
    """
    Converts a positive integer (up to 999) to its Hebrew Gematria representation.
    Correctly handles special combinations like 15 (×˜"×•) and 16 (×˜"×–) to avoid Yud-Heh/Yud-Vav.
    Formats multi-letter strings with double quotes before the last character.
    """
    if num <= 0 or num >= 1000:
        raise ValueError("Gematria conversion is only supported for integers between 1 and 999.")

    units = ["", "×", "×‘", "×’", "×“", "×”", "×•", "×–", "×—", "×˜"]
    tens = ["", "×™", "×›", "×œ", "×ž", "× ", "×¡", "×¢", "×¤", "×¦"]
    hundreds = ["", "×§", "×¨", "×©", "×ª", "×ª×§", "×ª×¨", "×ª×©", "×ª×ª", "×ª×ª×§"]

    h = num // 100
    t = (num % 100) // 10
    u = num % 10

    # Special cases for 15 and 16
    if t == 1 and u == 5:
        res = hundreds[h] + "×˜×•"
    elif t == 1 and u == 6:
        res = hundreds[h] + "×˜×–"
    else:
        res = hundreds[h] + tens[t] + units[u]

    # Format with quotes
    if len(res) > 1:
        return res[:-1] + '"' + res[-1]
    elif len(res) == 1:
        return res + "'"
    return ""


LETTER_NAMES = {
    "×": "××œ×£",
    "×‘": "×‘×™×ª",
    "×’": "×’×™×ž×œ",
    "×“": "×“×œ×ª",
    "×”": "×”×",
    "×•": "×•×•",
    "×–": "×–×™×Ÿ",
    "×—": "×—×™×ª",
    "×˜": "×˜×™×ª",
    "×™": "×™×•×“",
    "×›": "×›×£",
    "×œ": "×œ×ž×“",
    "×ž": "×ž×",
    "× ": "× ×•×Ÿ",
    "×¡": "×¡×ž×š",
    "×¢": "×¢×™×Ÿ",
    "×¤": "×¤×",
    "×¦": "×¦×“×™×§",
    "×§": "×§×•×£",
    "×¨": "×¨×™×©",
    "×©": "×©×™×Ÿ",
    "×ª": "×ª×•",
}


def int_to_spoken_gematria(num: int) -> str:
    """
    Phonetic letter-name form for TTS (e.g. 94 -> '×¦×“×™×§ ×“×œ×ª').
    Uses the same letter sequence as int_to_gematria, without quote marks.
    """
    written = int_to_gematria(num)
    letters = [ch for ch in written if ch in LETTER_NAMES]
    if not letters:
        raise ValueError(f"Could not derive spoken gematria for {num}.")
    return " ".join(LETTER_NAMES[ch] for ch in letters)

