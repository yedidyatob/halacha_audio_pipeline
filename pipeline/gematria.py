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


def gematria_to_int(text: str) -> int:
    """Inverse of int_to_gematria (1-999); geresh / gershayim (ASCII or Hebrew) are optional."""
    letters = text.strip().translate(str.maketrans("", "", "'\"\u05f3\u05f4"))
    for num in range(1, 1000):
        if int_to_gematria(num).translate(str.maketrans("", "", "'\"")) == letters:
            return num
    raise ValueError(f"Not a Hebrew number: '{text}'")
