"""LLM wrapper: request shape, refusal/truncation handling, per-call cost logging."""

from types import SimpleNamespace

import pytest

from niche_miner import db
from niche_miner.llm import LLM, LLMError
from niche_miner.settings import LLMSettings


class FakeMessages:
    def __init__(self, reply):
        self.reply = reply
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.reply


def fake_client(text="{}", stop_reason="end_turn", input_tokens=1000, output_tokens=500):
    reply = SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        stop_details=SimpleNamespace(category="cyber") if stop_reason == "refusal" else None,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens,
                              cache_creation_input_tokens=0, cache_read_input_tokens=0))
    return SimpleNamespace(messages=FakeMessages(reply))


def test_text_blocks_are_returned_and_cost_is_logged(conn):
    settings = LLMSettings(pricing={"m": {"input": 0.10, "output": 0.50}})
    llm = LLM(conn, settings, "run-1",
              client=fake_client('{"ok": true}', input_tokens=10_000, output_tokens=2_000))
    assert llm.complete(purpose="extract", model="m", system="s", user="u", n_items=20) == \
        '{"ok": true}'
    (usage,) = db.llm_usage(conn, "run-1")
    assert usage["calls"] == 1 and usage["input_tokens"] == 10_000
    assert usage["cost_usd"] == pytest.approx((10_000 * 0.10 + 2_000 * 0.50) / 1e6)


def test_request_shape_uses_structured_output_and_effort(conn):
    client = fake_client()
    LLM(conn, LLMSettings(effort="low"), "r", client=client).complete(
        purpose="extract", model="m", system="sys", user="hola", schema={"type": "object"})
    kwargs = client.messages.kwargs
    assert kwargs["output_config"] == {
        "effort": "low", "format": {"type": "json_schema", "schema": {"type": "object"}}}
    assert kwargs["system"] == "sys"
    assert kwargs["messages"] == [{"role": "user", "content": "hola"}]
    assert "temperature" not in kwargs  # rejected by current models


def test_effort_can_be_raised_for_one_call(conn):
    client = fake_client()
    LLM(conn, LLMSettings(effort="low"), "r", client=client).complete(
        purpose="themes", model="m", system="s", user="u", effort="medium")
    assert client.messages.kwargs["output_config"] == {"effort": "medium"}


def test_effort_can_be_disabled_for_older_models(conn):
    client = fake_client()
    LLM(conn, LLMSettings(effort=None), "r", client=client).complete(
        purpose="label", model="claude-haiku-4-5", system="s", user="u")
    assert "output_config" not in client.messages.kwargs


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_unusable_responses_raise_but_are_still_billed(conn, stop_reason):
    llm = LLM(conn, LLMSettings(), "r", client=fake_client(stop_reason=stop_reason))
    with pytest.raises(LLMError):
        llm.complete(purpose="extract", model="claude-haiku-5-5", system="s", user="u")
    assert db.llm_usage(conn, "r")[0]["calls"] == 1


def test_unpriced_model_is_logged_with_null_cost(conn):
    LLM(conn, LLMSettings(pricing={}), "r", client=fake_client()).complete(
        purpose="label", model="mystery-model", system="s", user="u")
    (usage,) = db.llm_usage(conn, "r")
    assert (usage["unpriced"], usage["cost_usd"]) == (1, 0)
