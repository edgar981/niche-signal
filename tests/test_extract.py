"""Extraction schema validation, verbatim spend check, and the batch/retry flow (fake LLM)."""

import pytest
from helpers import FakeLLM, echo_valid, good_result, make_signal, prompt_items, response

from niche_miner import db
from niche_miner.extract import Extractor, build_prompt, is_verbatim, parse_results, source_hint
from niche_miner.llm import LLMError
from niche_miner.models import ExtractedPain

# --- schema validation -------------------------------------------------------------------


def test_valid_result_parses():
    ok, errors = parse_results(response(good_result(0)), 1)
    assert errors == {}
    assert ok[0] == ExtractedPain(is_complaint=True, pain="Necesita resolver el problema 0.",
                                  pain_type="missing_capability", who="contador independiente",
                                  who_type="business", country="co", current_tool="Siigo",
                                  spend_signal=None, severity=3)


def test_complaint_needs_a_pain_type_and_known_enums():
    ok, errors = parse_results(response(good_result(0, pain_type=None),
                                        good_result(1, pain_type="bugs"),
                                        good_result(2, country="br")), 3)
    assert ok == {}
    assert "pain_type is missing" in errors[0]
    assert "pain_type" in errors[1] and "country" in errors[2]


def test_non_complaint_drops_pain_type():
    pain = ExtractedPain.model_validate(
        good_result(0, is_complaint=False, pain=None, pain_type="pricing", severity=1))
    assert pain.pain_type is None


@pytest.mark.parametrize(("who", "kept"), [
    ("usuario de software contable", None),
    ("Usuaria de Siigo Nube", None),
    ("un usuario", None),
    ("contador independiente", "contador independiente"),
    ("dueña de ferretería", "dueña de ferretería"),
])
def test_generic_who_is_dropped(who, kept):
    assert ExtractedPain.model_validate(good_result(0, who=who)).who == kept


@pytest.mark.parametrize(("tool", "kept"), [
    ("co.alegra.app", None),
    ("com.siigo.nubeapp", None),
    ("Siigo Nube", "Siigo Nube"),
    ("manual/Excel/WhatsApp", "manual/Excel/WhatsApp"),
])
def test_package_ids_are_not_tool_names(tool, kept):
    assert ExtractedPain.model_validate(good_result(0, current_tool=tool)).current_tool == kept


@pytest.mark.parametrize(("country", "excluded"), [
    ("co", 0), ("unknown", 0), ("mx", 1), ("pe", 1), ("es", 1), ("other", 1)])
def test_known_non_colombian_pains_are_flagged(conn, settings, country, excluded):
    db.insert_signals(conn, [make_signal(0)])
    Extractor(conn, FakeLLM(response(good_result(0, country=country))), settings.llm).run()
    row = stored(conn)[0]
    assert (row["country"], row["exclude_from_scoring"]) == (country, excluded)


def test_each_item_is_validated_independently():
    text = response(good_result(0), good_result(1, severity=9), good_result(2, pain=None),
                    good_result(3, who_type="empresa"), {"id": 4, "is_complaint": "tal vez"})
    ok, errors = parse_results(text, 6)
    assert list(ok) == [0]
    assert "severity" in errors[1]
    assert "pain is empty" in errors[2]
    assert "who_type" in errors[3]
    assert "is_complaint" in errors[4]
    assert errors[5] == "missing from response"


def test_non_complaint_needs_no_pain():
    ok, _ = parse_results(response(good_result(0, is_complaint=False, pain=None, severity=1)), 1)
    assert ok[0].is_complaint is False and ok[0].pain is None


def test_nullish_strings_become_none():
    pain = ExtractedPain.model_validate(
        good_result(0, current_tool="null", spend_signal="  ", who="Desconocido"))
    assert (pain.current_tool, pain.spend_signal, pain.who) == (None, None, None)


def test_out_of_batch_and_repeated_ids_are_ignored():
    ok, errors = parse_results(response(good_result(0), good_result(0, severity=1),
                                        good_result(7), good_result(True)), 2)
    assert ok[0].severity == 3  # first valid result for an id wins
    assert errors == {1: "missing from response"}


@pytest.mark.parametrize("bad", ["lo siento, no puedo", "[1, 2", '{"otra_cosa": 1}'])
def test_unparseable_response_raises(bad):
    with pytest.raises(ValueError):
        parse_results(bad, 1)


# --- verbatim spend quotes -----------------------------------------------------------------

SPEND_TEXT = "Al final le pagué 200 mil a un contador para que me arreglara la factura."


@pytest.mark.parametrize(("quote", "expected"), [
    ("le pagué 200 mil a un contador", True),
    ("Le  pagué 200 MIL a un contador.", True),     # case, spacing, trailing punctuation
    ("“le pagué 200 mil”", True),                    # surrounding quotes
    ("le pagué... a un contador", True),             # ellipsis-joined fragments
    ("le pagué 300 mil a un contador", False),       # altered amount
    ("contraté a un asistente", False),              # invented
    ("", False),
])
def test_is_verbatim(quote, expected):
    assert is_verbatim(quote, SPEND_TEXT) is expected


# --- prompt --------------------------------------------------------------------------------


def test_prompt_uses_batch_local_ids_and_bounds_text(conn):
    db.insert_signals(conn, [make_signal(i) for i in range(2)])
    items = prompt_items({"user": build_prompt(db.pending_signals(conn), max_chars=20)})
    assert [item["id"] for item in items] == [0, 1]
    assert all(len(item["texto"]) <= 20 for item in items)


def test_source_hint_names_the_reviewed_app():
    url = "https://play.google.com/store/apps/details?id=com.siigo.nubeapp&hl=es&reviewId=x"
    assert "com.siigo.nubeapp" in source_hint("google_play", url)
    assert source_hint("youtube", None) == "youtube"


# --- extractor flow ------------------------------------------------------------------------


def stored(conn):
    return conn.execute("SELECT * FROM extracted_pains ORDER BY raw_signal_id").fetchall()


def test_signals_are_sent_in_batches_of_twenty(conn, settings):
    db.insert_signals(conn, [make_signal(i) for i in range(45)])
    llm = FakeLLM(echo_valid, echo_valid, echo_valid)
    stats = Extractor(conn, llm, settings.llm).run()
    assert [len(prompt_items(call)) for call in llm.calls] == [20, 20, 5]
    assert (stats.signals, stats.complaints, stats.failed) == (45, 45, 0)
    assert all(call["model"] == "claude-haiku-5-5" for call in llm.calls)


def test_failed_items_are_retried_once_then_marked_failed(conn, settings):
    db.insert_signals(conn, [make_signal(i) for i in range(3)])
    llm = FakeLLM(
        response(good_result(0), good_result(1, severity=0)),  # 1 invalid, 2 missing
        response(good_result(0)),                               # retry of [1, 2]: only 1 fixed
    )
    stats = Extractor(conn, llm, settings.llm).run()
    assert [len(prompt_items(call)) for call in llm.calls] == [3, 2]
    assert (stats.signals, stats.failed, stats.retried) == (3, 1, 2)
    rows = stored(conn)
    assert [r["error"] for r in rows] == [None, None, "missing from response"]
    assert rows[1]["pain"] == "Necesita resolver el problema 0."  # retry result mapped back
    assert rows[2]["is_complaint"] is None


def test_unparseable_batch_is_retried_whole(conn, settings):
    db.insert_signals(conn, [make_signal(i) for i in range(2)])
    stats = Extractor(conn, FakeLLM("lo siento, no puedo", echo_valid), settings.llm).run()
    assert stats.failed == 0 and stats.retried == 2


def test_refused_batch_is_marked_failed_then_retried_on_request(conn, settings):
    db.insert_signals(conn, [make_signal(i) for i in range(2)])
    refusal = LLMError("refused (general_harms)")
    stats = Extractor(conn, FakeLLM(refusal, refusal), settings.llm).run()
    assert stats.failed == 2
    assert all("refused" in r["error"] for r in stored(conn))
    stats = Extractor(conn, FakeLLM(echo_valid), settings.llm).run(retry_failed=True)
    assert (stats.signals, stats.failed) == (2, 0)
    assert all(r["error"] is None for r in stored(conn))


def test_already_extracted_signals_are_not_resent(conn, settings):
    db.insert_signals(conn, [make_signal(0)])
    Extractor(conn, FakeLLM(echo_valid), settings.llm).run()
    llm = FakeLLM()
    assert Extractor(conn, llm, settings.llm).run().signals == 0
    assert llm.calls == []


def test_redo_re_extracts_everything(conn, settings):
    db.insert_signals(conn, [make_signal(i) for i in range(3)])
    Extractor(conn, FakeLLM(echo_valid), settings.llm).run()
    llm = FakeLLM(lambda call: response(*(good_result(it["id"], pain_type="pricing")
                                          for it in prompt_items(call))))
    stats = Extractor(conn, llm, settings.llm).run(redo=True)
    assert stats.signals == 3 and len(llm.calls) == 1
    assert {r["pain_type"] for r in stored(conn)} == {"pricing"}


def test_non_verbatim_spend_quote_is_dropped(conn, settings):
    text = "Le pago 300 mil al mes a un contador porque la app no concilia los bancos."
    db.insert_signals(conn, [make_signal(0, text=text), make_signal(1, text=text + " Otra vez.")])
    llm = FakeLLM(response(good_result(0, spend_signal="le pago 300 mil al mes a un contador"),
                           good_result(1, spend_signal="gasto dos millones al año")))
    stats = Extractor(conn, llm, settings.llm).run()
    assert [r["spend_signal"] for r in stored(conn)] == ["le pago 300 mil al mes a un contador",
                                                          None]
    assert stats.spend_dropped == 1
