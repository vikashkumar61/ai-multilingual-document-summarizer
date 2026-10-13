import re

_CJK_CHARACTERS = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\u00ad", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def count_words(text: str) -> int:
    cjk_characters = len(_CJK_CHARACTERS.findall(text))
    remaining_text = _CJK_CHARACTERS.sub(" ", text)
    return len(re.findall(r"\w+", remaining_text, flags=re.UNICODE)) + cjk_characters
