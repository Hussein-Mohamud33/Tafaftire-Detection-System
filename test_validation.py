
import re

def is_readable_news(text):
    if not isinstance(text, str):
        return False

    text = text.strip()
    if not text:
        return False

    words = text.split()
    if not words:
        return False

    # Check for extremely long words
    max_word_len = max(len(w) for w in words)
    if max_word_len > 20:
        return False

    # Check for long consonant sequences (e.g., "trqpw")
    if re.search(r'[^aeiouy ]{6,}', text.lower()):
        return False

    # Check for character repetition
    if re.search(r'(.)\1{8,}', text):
        return False

    # Vowel ratio check
    letters = re.findall(r"[a-zA-Z]", text.lower())
    if len(letters) == 0:
        return False

    vowels = sum(1 for c in letters if c in "aeiou")
    ratio = vowels / len(letters)

    if ratio < 0.20 or ratio > 0.70:
        return False

    return True

# Test cases
test_inputs = [
    "sghajdbwegyriqwhjdhweajdbjhwgdyu guigwquiguiweguieg", # Gibberish (long words)
    "aaaaaaaaaaaaaaaaaaaa", # Repetitive chars
    "Normal Somali news text should pass here.", # Valid
    "Kani waa warkii caalamka ee saaka.", # Valid Somali
]

for t in test_inputs:
    print(f"Input: {t[:30]}... -> Readable: {is_readable_news(t)}")
