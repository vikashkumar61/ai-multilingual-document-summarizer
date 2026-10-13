import torch

from summarizer import MultilingualSummarizer


class WordTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(token_ids)

    def build_inputs_with_special_tokens(self, token_ids):
        return token_ids


class BatchTokenizer:
    def __call__(self, texts, **kwargs):
        return {
            "input_ids": torch.ones((len(texts), 2), dtype=torch.long),
            "attention_mask": torch.ones((len(texts), 2), dtype=torch.long),
        }

    def batch_decode(self, output, skip_special_tokens=True):
        return [f"meaningful summary {index}" for index in range(len(output))]

    def build_inputs_with_special_tokens(self, token_ids):
        return [1, *token_ids, 2]

    def pad(self, features, padding=True, return_tensors="pt"):
        width = max(len(feature["input_ids"]) for feature in features)
        input_ids = torch.zeros((len(features), width), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for index, feature in enumerate(features):
            size = len(feature["input_ids"])
            input_ids[index, :size] = torch.tensor(feature["input_ids"])
            attention_mask[index, :size] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def test_short_text_is_summarized_directly_without_chunking():
    summarizer = MultilingualSummarizer.__new__(MultilingualSummarizer)
    summarizer.tokenizer = WordTokenizer()
    summarizer.device = torch.device("cpu")
    batches = []

    def fake_summarize_token_chunks(token_chunks, max_length=128, min_length=16):
        batches.append((len(token_chunks), max_length, min_length))
        return [("word " * 120).strip() for _ in token_chunks]

    summarizer.summarize_token_chunks = fake_summarize_token_chunks

    summary, stats = summarizer.summarize_document(
        " ".join(f"token{index}" for index in range(500)),
        target_words=40,
        sentences_per_chunk=2,
    )

    assert batches == [(1, 60, 8)]
    assert stats["chunks"] == 1
    assert summary.strip()


def test_long_text_gets_chunk_summaries_then_a_final_summary():
    summarizer = MultilingualSummarizer.__new__(MultilingualSummarizer)
    summarizer.tokenizer = WordTokenizer()
    summarizer.device = torch.device("cpu")
    calls = []

    def fake_summarize_token_chunks(token_chunks, max_length=128, min_length=16):
        calls.append((len(token_chunks), max_length, min_length))
        return [("word " * 10).strip() for _ in token_chunks]

    summarizer.summarize_token_chunks = fake_summarize_token_chunks

    summary, stats = summarizer.summarize_document(
        " ".join(f"token{index}" for index in range(1200)),
        target_words=40,
        max_input_tokens=400,
    )

    assert calls == [(1, 96, 12), (1, 96, 12), (1, 96, 12), (1, 60, 8)]
    assert stats["chunks"] == 3
    assert summary


def test_batch_summarization_returns_one_summary_per_input():
    summarizer = MultilingualSummarizer.__new__(MultilingualSummarizer)
    summarizer.tokenizer = BatchTokenizer()
    summarizer.device = torch.device("cpu")

    class FakeModel:
        def generate(self, **kwargs):
            assert kwargs["num_beams"] == 2
            assert kwargs["do_sample"] is False
            assert torch.is_inference_mode_enabled()
            return torch.ones((kwargs["input_ids"].shape[0], 2), dtype=torch.long)

    summarizer.model = FakeModel()

    assert summarizer.summarize_chunks(["first chunk", "second chunk"]) == [
        "meaningful summary 0",
        "meaningful summary 1",
    ]


def test_token_chunk_batch_preserves_pre_tokenized_inputs():
    summarizer = MultilingualSummarizer.__new__(MultilingualSummarizer)
    summarizer.tokenizer = BatchTokenizer()
    summarizer.device = torch.device("cpu")

    class FakeModel:
        def generate(self, **kwargs):
            assert kwargs["input_ids"].shape == (2, 3)
            return torch.ones((2, 2), dtype=torch.long)

    summarizer.model = FakeModel()

    assert summarizer.summarize_token_chunks([[5], [6]]) == [
        "meaningful summary 0",
        "meaningful summary 1",
    ]


def test_long_document_uses_exactly_one_final_compression_generation():
    summarizer = MultilingualSummarizer.__new__(MultilingualSummarizer)
    summarizer.tokenizer = WordTokenizer()
    summarizer.device = torch.device("cpu")
    calls = []

    def fake_summarize_token_chunks(token_chunks, max_length=128, min_length=16):
        calls.append((len(token_chunks), max_length))
        return [("meaningful chunk summary " * 20) for _ in token_chunks]

    summarizer.summarize_token_chunks = fake_summarize_token_chunks
    summary, stats = summarizer.summarize_document(
        " ".join(f"token{index}" for index in range(1200)),
        target_words=40,
        max_input_tokens=400,
    )

    assert calls == [(1, 96), (1, 96), (1, 96), (1, 60)]
    assert stats["chunks"] == 3
    assert summary


def test_clean_summary_rejects_empty_and_too_short_output():
    for decoded in ("", "   \n ", "…", "one token"):
        try:
            MultilingualSummarizer._clean_summary(decoded)
        except ValueError as exc:
            assert "too-short" in str(exc)
        else:
            raise AssertionError(f"Expected invalid summary to be rejected: {decoded!r}")


def test_clean_summary_normalizes_meaningful_multilingual_text():
    assert MultilingualSummarizer._clean_summary("  ملخص   مفيد للنص  ") == (
        "ملخص مفيد للنص"
    )
    assert MultilingualSummarizer._clean_summary("यह एक सारांश है") == (
        "यह एक सारांश है"
    )


def test_chunking_uses_near_full_model_inputs_without_tiny_tail():
    summarizer = MultilingualSummarizer.__new__(MultilingualSummarizer)
    summarizer.tokenizer = WordTokenizer()

    chunks = summarizer._split_oversized_chunks(
        [" ".join(f"token{i}" for i in range(850))],
        max_input_tokens=100,
    )
    chunk_sizes = [len(chunk.split()) for chunk in chunks]

    assert len(chunks) == 9
    assert max(chunk_sizes) <= 100
    assert min(chunk_sizes) >= 32


def test_empty_input_returns_zero_chunk_statistics():
    summarizer = MultilingualSummarizer.__new__(MultilingualSummarizer)
    summarizer.tokenizer = WordTokenizer()

    summary, stats = summarizer.summarize_document("  ")

    assert summary == ""
    assert stats["chunks"] == 0


def test_sentence_chunking_recognizes_cjk_punctuation():
    chunks = MultilingualSummarizer.split_into_chunks(
        "这是第一句。第二句！第三句？",
        sentences_per_chunk=2,
    )

    assert chunks == ["这是第一句。第二句！", "第三句？"]
