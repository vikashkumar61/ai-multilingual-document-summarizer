from pathlib import Path

from streamlit.testing.v1 import AppTest

import summarizer


class FakeSummarizer:
    def summarize_document(self, text, **kwargs):
        return "This is a meaningful test summary.", {
            "original_words": 40,
            "summary_words": 6,
            "reduction_percent": 85.0,
            "chunks": 1,
        }


def test_generate_summary_populates_session_state_and_renders_result(monkeypatch):
    monkeypatch.setattr(summarizer, "MultilingualSummarizer", FakeSummarizer)
    app_file = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_file)).run(timeout=15)

    app.text_area[0].set_value(
        "This test document contains enough words to enable summary generation. "
        "It verifies that a generated summary is stored and rendered correctly."
    )
    app.run(timeout=15)
    app.button[0].click()
    app.run(timeout=15)

    result = app.session_state["summary_result"]
    assert result["summary"] == "This is a meaningful test summary."
    assert result["stats"]["summary_words"] == 6
    assert any(
        "This is a meaningful test summary." in element.value
        for element in app.markdown
    )
