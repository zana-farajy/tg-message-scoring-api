import re

def normalize_text(text: str) -> str:
    """
    Normalizes Persian and English text.
    1. Converts English letters to lowercase.
    2. Translates Arabic characters 'ي' and 'ك' to Persian 'ی' and 'ک'.
    3. Cleans up ZWNJ (نیم‌فاصله) - removes multiple ZWNJs and spaces around them.
    4. Removes Arabic/Persian diacritics (Fatha, Damma, Kasra, Shadda, Sukun, Tanween, Kashida).
    5. Replaces multiple consecutive whitespaces with a single space.
    6. Strips leading and trailing whitespaces.
    """
    if not text:
        return ""
    
    # 1. Lowercase English letters
    text = text.lower()
    
    # 2. Convert Arabic characters to Persian equivalents
    text = text.replace("ي", "ی").replace("ك", "ک")
    
    # 3. Clean up ZWNJs (zero-width non-joiner, \u200c)
    # Remove consecutive ZWNJs
    text = re.sub(r'\u200c+', '\u200c', text)
    # Remove spaces around ZWNJ
    text = re.sub(r'\s*\u200c\s*', '\u200c', text)
    # Remove ZWNJ from start or end of the text
    text = re.sub(r'^\u200c|\u200c$', '', text)
    
    # 4. Remove Arabic/Persian diacritics & Kashida (\u0640)
    # Diacritics range: \u064b to \u0652
    text = re.sub(r'[\u064b-\u0652\u0640]', '', text)
    
    # 5. Remove decorative characters (e.g., repeating dashes, lines, stars)
    # but keep standard punctuation for word boundary matching
    text = re.sub(r'[░▒▓█▀▄■▪▬▲▼▶◀●🟪🔵🟢🔴🟡🔸🔹⭐✨💫⚠️📌📣📢🚀💎💡⚙️🔧🛠️💼💵💰📊📈📆📅📌📍👤👥📞📧✉️🌐🔗🔗]', ' ', text)
    
    # 6. Replace multiple spaces/newlines with a single space
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()
