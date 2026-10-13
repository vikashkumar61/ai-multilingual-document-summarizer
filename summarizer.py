import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from preprocessing import count_words

MODEL_NAME = "csebuetnlp/mT5_multilingual_XLSum"
MAX_INPUT_LENGTH = 512
INTERMEDIATE_MAX_LENGTH = 96
INTERMEDIATE_MIN_LENGTH = 12


class MultilingualSummarizer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def split_into_chunks(text, sentences_per_chunk=10):
        sentences = re.split(r"(?<=[.!?।。！？])\s*", text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        chunks = []
        cjk_boundary = re.compile(r"[\u3400-\u9fff\u3040-\u30ff\u3002\uff01\uff1f]$")
        cjk_start = re.compile(r"^[\u3400-\u9fff\u3040-\u30ff]")
        for start in range(0, len(sentences), sentences_per_chunk):
            group = sentences[start:start + sentences_per_chunk]
            chunk = group[0]
            for sentence in group[1:]:
                separator = (
                    ""
                    if cjk_boundary.search(chunk) and cjk_start.match(sentence)
                    else " "
                )
                chunk += separator + sentence
            chunks.append(chunk)
        return chunks

    def summarize_chunks(
        self,
        texts,
        max_length=128,
        min_length=16,
        max_input_length=MAX_INPUT_LENGTH,
    ):
        if not texts:
            return []

        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        num_beams = 1 if self.device.type == "cpu" else 2
        with torch.inference_mode():
            output = self.model.generate(
                **encoded,
                max_length=max_length,
                min_length=min(min_length, max_length),
                num_beams=num_beams,
                do_sample=False,
                no_repeat_ngram_size=3,
                repetition_penalty=1.15,
            )

        return self._decode_summaries(output)

    def summarize_token_chunks(
        self,
        token_chunks,
        max_length=128,
        min_length=16,
    ):
        if not token_chunks:
            return []

        features = [
            {
                "input_ids": self.tokenizer.build_inputs_with_special_tokens(
                    token_ids
                )
            }
            for token_ids in token_chunks
        ]
        encoded = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        num_beams = 1 if self.device.type == "cpu" else 2
        with torch.inference_mode():
            output = self.model.generate(
                **encoded,
                max_length=max_length,
                min_length=min(min_length, max_length),
                num_beams=num_beams,
                do_sample=False,
                no_repeat_ngram_size=3,
                repetition_penalty=1.15,
            )

        return self._decode_summaries(output)

    def summarize_chunk(
        self,
        text,
        max_length=128,
        min_length=16,
    ):
        return self.summarize_chunks(
            [text],
            max_length=max_length,
            min_length=min_length,
        )[0]

    @staticmethod
    def _split_token_ids(token_ids, max_input_tokens):
        chunks = [
            token_ids[start:start + max_input_tokens]
            for start in range(0, len(token_ids), max_input_tokens)
        ]
        if len(chunks) > 1 and len(chunks[-1]) < 32:
            transfer_count = 32 - len(chunks[-1])
            if len(chunks[-2]) - transfer_count >= 32:
                chunks[-1] = chunks[-2][-transfer_count:] + chunks[-1]
                chunks[-2] = chunks[-2][:-transfer_count]
        return [chunk for chunk in chunks if chunk]

    def _split_oversized_chunks(self, chunks, max_input_tokens=448):
        text = " ".join(chunk for chunk in chunks if chunk).strip()
        if not text:
            return []

        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return [
            self.tokenizer.decode(token_slice, skip_special_tokens=True).strip()
            for token_slice in self._split_token_ids(token_ids, max_input_tokens)
        ]

    @staticmethod
    def _clean_summary(summary):
        if not isinstance(summary, str):
            raise ValueError("The model returned a summary in an unexpected format.")

        cleaned = re.sub(r"\s+", " ", summary).strip()
        if count_words(cleaned) < 3:
            raise ValueError(
                "The model returned an empty or too-short summary. "
                "Try a longer document or generate the summary again."
            )
        return cleaned

    def _decode_summaries(self, output):
        decoded = self.tokenizer.batch_decode(
            output,
            skip_special_tokens=True,
        )
        if len(decoded) != len(output):
            raise ValueError(
                "The model returned a different number of summaries than inputs."
            )
        return [self._clean_summary(summary) for summary in decoded]

    def summarize_document(
        self,
        text,
        target_words=100,
        sentences_per_chunk=10,
        progress_callback=None,
        max_input_tokens=448,
        max_input_length=MAX_INPUT_LENGTH,
    ):
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            return "", {
                "original_words": 0,
                "summary_words": 0,
                "reduction_percent": 0.0,
                "chunks": 0,
            }

        special_tokens = len(self.tokenizer.build_inputs_with_special_tokens([]))
        safe_input_tokens = max(1, max_input_length - special_tokens)
        original_words = count_words(text)

        if original_words <= 500 and len(token_ids) <= safe_input_tokens:
            token_chunks = [token_ids]
        else:
            chunk_limit = (
                safe_input_tokens
                if original_words <= 500
                else min(max_input_tokens, safe_input_tokens)
            )
            token_chunks = self._split_token_ids(token_ids, chunk_limit)

        initial_chunk_count = len(token_chunks)
        final_max_length = max(48, min(160, int(target_words * 1.5)))
        final_min_length = min(24, max(8, int(target_words * 0.2)))
        batch_size = 1 if self.device.type == "cpu" else 4

        if initial_chunk_count == 1:
            result = self.summarize_token_chunks(
                token_chunks,
                max_length=final_max_length,
                min_length=final_min_length,
            )
            if len(result) != 1:
                raise ValueError("The model did not return a summary for the document.")
            combined = self._clean_summary(result[0])
            if progress_callback:
                progress_callback(99)
        else:
            partial = []
            for start in range(0, len(token_chunks), batch_size):
                batch = token_chunks[start:start + batch_size]
                partial.extend(
                    self.summarize_token_chunks(
                        batch,
                        max_length=INTERMEDIATE_MAX_LENGTH,
                        min_length=INTERMEDIATE_MIN_LENGTH,
                    )
                )
                if progress_callback:
                    progress_callback(
                        (start + len(batch)) / len(token_chunks) * 70
                    )

            if len(partial) != len(token_chunks):
                raise ValueError(
                    "The model did not return a summary for every document chunk."
                )

            combined_ids = self.tokenizer.encode(
                " ".join(partial),
                add_special_tokens=False,
            )[:safe_input_tokens]
            if not combined_ids:
                raise ValueError(
                    "The chunk summaries did not contain text for final summarization."
                )
            final_result = self.summarize_token_chunks(
                [combined_ids],
                max_length=final_max_length,
                min_length=final_min_length,
            )
            if len(final_result) != 1:
                raise ValueError("The model did not return a final document summary.")
            combined = self._clean_summary(final_result[0])
            if progress_callback:
                progress_callback(99)

        summary_words = count_words(combined)
        reduction = (
            max(0.0, (1 - summary_words / original_words) * 100)
            if original_words else 0.0
        )

        return combined, {
            "original_words": original_words,
            "summary_words": summary_words,
            "reduction_percent": reduction,
            "chunks": initial_chunk_count,
        }
