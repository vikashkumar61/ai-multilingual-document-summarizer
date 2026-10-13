from preprocessing import clean_text, count_words


def test_clean_text_removes_control_characters_and_soft_hyphens():
    assert clean_text("multi\u00adlingual\x00  text\r\nhere") == "multilingual text\nhere"


def test_count_words_counts_cjk_characters_for_length_feedback():
    assert count_words("你好世界") == 4
