# A comprehensive dictionary of Halachic abbreviations and their precise TTS equivalents
RABBINIC_NIKKUD_DICT = {
    # === Rishonim (ראשונים) ===
    'רש"י': 'רַשִׁי',
    'רמב"ם': 'רַמְבַּם',
    'רמב"ן': 'רַמְבַּן',
    'רשב"א': 'רַשְׁבָּא',
    'ריטב"א': 'רִיטְבָּא',
    'רשב"ם': 'רַשְׁבַּם',
    'ראב"ד': 'רַאֲבַד', # Fixed: בלי דגש בב'
    'רא"ש': 'רֹאשׁ',
    'רי"ף': 'רִיף',
    'ר"ת': 'רַבֵּינוּ תַּם', # Fixed: נפתח למילה מלאה
    'ר"ן': 'רַן',
    'רז"ה': 'בַּעַל הַמָּאוֹר', # Fixed: הוחלף לשם הספר
    'ריב"ש': 'רִיבָשׁ', # Fixed: בלי דגש בב'
    'תשב"ץ': 'תַּשְׁבֵּץ',
    'רד"ק': 'רַדַק',
    'רלב"ג': 'רַלְבַּג',
    'רמ"ה': 'יַד רָמָה', # Fixed: הוחלף לשם הספר
    'סמ"ג': 'סְמַג',
    'סמ"ק': 'סְמַק',
    'רמ"ך': 'רָמָךְ',
    
    # === Acharonim (אחרונים) ===
    'רמ"א': 'רַמָא',
    'ש"ך': 'שַׁךְ',
    'ט"ז': 'טַז',
    'סמ"ע': 'סְמַע',
    'ב"ח': 'בַּח',
    'מג"א': 'מָגֵן אַבְרָהָם', # Fixed: נפתח
    'מהרש"א': 'מַהַרְשָׁא',
    'מהרש"ל': 'מַהַרְשַׁל',
    'מהר"ם': 'מַהֲרַם',
    'מהרי"ל': 'מַהֲרִיל',
    'מהרי"ט': 'מַהֲרִיט',
    'רדב"ז': 'רַדְבַּז',
    'חיד"א': 'חִידָא',
    'חת"ס': 'חֲתַם סוֹפֵר', # Fixed: נפתח
    'גר"א': 'גְּרָא',
    'נצי"ב': 'נָצִיב',
    'חזו"א': 'חֲזוֹן אִישׁ', # Fixed: נפתח
    'פרמ"ג': 'פְּרִי מְגָדִים', # Fixed: נפתח
    'נוב"י': 'נוֹדָע בִּיהוּדָה', # Fixed: נפתח
    'כה"ח': 'כַּף הַחַיִּים', # Fixed: נפתח
    'משנ"ב': 'מִשְׁנָה בְּרוּרָה', # Fixed: נפתח
    'ערוה"ש': 'עֲרוּךְ הַשֻּׁלְחָן', # Fixed: נפתח
    
    # === Major Halachic Works & Terms ===
    'שו"ע': 'שׁוּלְחָן עָרוּךְ', # Fixed: נפתח
    'ב"י': 'בֵּית יוֹסֵף', # Fixed: נפתח
    'יו"ד': 'יוֹרֶה דֵּעָה', # Fixed: נפתח
    'או"ח': 'אוֹרַח חַיִּים', # Fixed: נפתח
    'אבן העזר': 'אֶבֶן הָעֵזֶר',
    'אה"ע': 'אֶבֶן הָעֵזֶר', # Fixed: נפתח
    'חו"מ': 'חוֹשֶׁן מִשְׁפָּט', # Fixed: נפתח
    'חנ"ן': 'חֲנָן', 
    'נ"ט': 'נַט', # Fixed: הושאר כקיצור עם ניקוד קריא
    'ע"פ': 'עַל פִּי',
    'אע"פ': 'אַף עַל פִּי',
    'שו"ת': 'שׁוּ"ת'
}

def apply_nikkud_to_abbreviations(text: str) -> str:
    """
    Scans the text and replaces Rabbinic acronyms with their Nikkud/expanded counterparts.
    Keys are sorted by length descending to prevent partial word replacements.
    """
    # Sort the dictionary keys from longest to shortest
    sorted_keys = sorted(RABBINIC_NIKKUD_DICT.keys(), key=len, reverse=True)
    
    # Replace each occurrence in the text
    for key in sorted_keys:
        text = text.replace(key, RABBINIC_NIKKUD_DICT[key])
        
    return text