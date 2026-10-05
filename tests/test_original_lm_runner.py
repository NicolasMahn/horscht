import json

import pytest

from evaluation import original_lm_runner as runner


class SyntheticTokenizer:
    chat_template = "synthetic-chat-template-v1"

    class _Backend:
        @staticmethod
        def to_str():
            return "synthetic-tokenizer-v1"

    backend_tokenizer = _Backend()

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is False
        assert add_generation_prompt is True
        assert enable_thinking is False
        return f"<system>{messages[0]['content']}</system><user>{messages[1]['content']}</user><assistant>"

    def __call__(self, texts, *, add_special_tokens, truncation, padding):
        assert add_special_tokens is False
        assert truncation is False
        assert padding is False
        if isinstance(texts, str):
            texts = [texts]
        return {"input_ids": [[ord(char) for char in text] for text in texts]}


class UnstableLabelTokenizer(SyntheticTokenizer):
    def __call__(self, texts, *, add_special_tokens, truncation, padding):
        if isinstance(texts, str):
            texts = [texts]
        rows = []
        for text in texts:
            values = [ord(char) for char in text]
            if text.endswith("A"):
                values[-2:] = [999999]
            rows.append(values)
        return {"input_ids": rows}


def _protocol():
    return {
        "prompt": {
            "system_message": "Choose one answer letter only.",
        },
        "inference": {"max_input_tokens": 32768},
        "canonical_item_count": 1,
        "expected_group_counts": {"mmlu-pro-test": 1},
    }


def _synthetic_row():
    return {
        "id": "synthetic-id",
        "context": "A made-up blue shape.",
        "question": "Which shape is described?",
        "options": ["A made-up triangle", "A made-up circle", "A made-up square"],
    }


def test_chat_prompt_passes_only_allowed_synthetic_fields():
    tokenizer = SyntheticTokenizer()
    prompt = runner._chat_prompt(tokenizer, _synthetic_row(), _protocol())

    assert "A made-up blue shape." in prompt
    assert "Which shape is described?" in prompt
    assert "A. A made-up triangle" in prompt
    assert "B. A made-up circle" in prompt
    assert "C. A made-up square" in prompt
    assert "synthetic-id" not in prompt


def test_candidate_letters_are_single_prefix_stable_tokens():
    tokenizer = SyntheticTokenizer()
    prompt = runner._chat_prompt(tokenizer, _synthetic_row(), _protocol())
    input_ids, label_ids = runner._prompt_and_label_ids(tokenizer, prompt, 3)

    assert len(label_ids) == 3
    assert len(set(label_ids)) == 3
    assert input_ids[-1] == ord(">")


def test_candidate_letter_tokenization_rejects_prefix_merging():
    tokenizer = UnstableLabelTokenizer()
    prompt = runner._chat_prompt(tokenizer, _synthetic_row(), _protocol())

    with pytest.raises(runner.OriginalEvaluationError, match="choice_label_not_single_prefix_stable_token"):
        runner._prompt_and_label_ids(tokenizer, prompt, 3)


def test_restricted_logits_ignore_noncandidate_vocabulary_scores():
    labels = [8, 3, 6]
    logits = [10000.0, -5.0, 0.0, 1.25, 20000.0, 2.0, 2.5, -10000.0, 3.0]

    assert runner._restricted_choice_logits(logits, labels, 3) == [3.0, 1.25, 2.5]


@pytest.mark.parametrize(
    ("choice_count", "expected"),
    [(2, "2-4"), (4, "2-4"), (5, "5-8"), (8, "5-8"), (9, "9-16"), (16, "9-16"), (17, "17+")],
)
def test_choice_count_diagnostic_bins(choice_count, expected):
    assert runner._choice_bin(choice_count) == expected


@pytest.mark.parametrize(
    ("token_count", "expected"),
    [(1, "<=256"), (256, "<=256"), (257, "257-512"), (512, "257-512"),
     (513, "513-1024"), (1024, "513-1024"), (1025, "1025-2048"),
     (2048, "1025-2048"), (2049, ">2048")],
)
def test_input_length_diagnostic_bins(token_count, expected):
    assert runner._token_bin(token_count) == expected


def test_synthetic_preflight_validates_label_contexts_and_counts(tmp_path):
    item_path = tmp_path / "synthetic-items.jsonl"
    item_path.write_text(json.dumps({
        "item_id": "synthetic-id",
        "context": "A made-up blue shape.",
        "question": "Which shape is described?",
        "options": ["A made-up triangle", "A made-up circle", "A made-up square"],
        "answer_index": 2,
    }) + "\n", encoding="utf-8")

    result = runner._preflight_inputs(item_path, SyntheticTokenizer(), _protocol())

    assert result["items_checked"] == 1
    assert result["groups_checked"] == 1
    assert result["label_contexts_checked"] == 1
    assert result["over_limit_items"] == 0
    assert result["max_input_tokens"] > 1


def test_cursor_persists_only_sufficient_statistics(tmp_path):
    state = runner._empty_cursor("a" * 64, "b" * 64, 1)
    cursor_path = tmp_path / "cursor.json"
    runner._write_cursor(cursor_path, state)
    text = cursor_path.read_text(encoding="utf-8")
    stored = json.loads(text)

    assert "synthetic-id" not in text
    assert "answer_index" not in text
    assert "prediction" not in text
    assert stored["next_item_index"] == 0
    assert set(stored["by_source_slice"]) == set()
