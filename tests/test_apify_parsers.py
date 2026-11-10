"""Apify parsers on synthetic actor output (made-up people and texts)."""

import json
from datetime import UTC, datetime
from pathlib import Path

from niche_miner.collectors.apify_common import render, usable_id
from niche_miner.collectors.apify_facebook import parse_posts
from niche_miner.collectors.apify_gmaps import parse_places
from niche_miner.collectors.apify_tiktok import parse_comments, parse_videos, video_key
from niche_miner.collectors.base import hash_author

FIXTURES = Path(__file__).parent / "fixtures"


def load(name: str):
    return json.loads((FIXTURES / f"apify_{name}.json").read_text("utf-8"))


GMAPS, VIDEOS, COMMENTS, POSTS = (load(n) for n in (
    "gmaps_places", "tiktok_videos", "tiktok_comments", "facebook_posts"))
# Commenter / reviewer / poster identities in the fixtures, none of which may be stored.
PEOPLE = ("Ana Ejemplo", "Carlos Ejemplo", "María Ejemplo", "maria.ejemplo", "Pedro Ejemplo",
          "usuario.ejemplo1", "usuario.ejemplo2", "310 555 0199", "carlos@ejemplo.com",
          "soporte@contables.example", "avatar1")


def test_render_placeholders():
    config = {"q": ["{target}"], "url": "https://x.com/{target}/feed", "urls": "{video_urls}",
              "nested": [{"url": "{target}"}], "n": 5, "other": "{unknown}"}
    assert render(config, {"target": "abc", "video_urls": ["u1", "u2"]}) == {
        "q": ["abc"], "url": "https://x.com/abc/feed", "urls": ["u1", "u2"],
        "nested": [{"url": "abc"}], "n": 5, "other": "{unknown}"}


def test_masked_ids_are_not_ids():
    assert usable_id("******") is None and usable_id("  ") is None and usable_id(None) is None
    assert usable_id(" 123 ") == "123"


def test_gmaps_keeps_low_star_reviews_with_text():
    pairs = list(parse_places(GMAPS, "n", max_stars=3))
    signals = [s for s, _ in pairs]
    assert [s.external_id for s in signals] == [
        "RevFakeLow0001", "RevFakeMid0002", "RevFakeShort0005", "RevFakeLow0006"]
    first, mid, _, masked = signals
    assert first.author_hash == hash_author("apify_gmaps", "reviewer:111111111111111111111")
    assert mid.author_hash == hash_author("apify_gmaps", "review:RevFakeMid0002")
    assert masked.author_hash == hash_author("apify_gmaps", "review:RevFakeLow0006")
    assert first.text.endswith("Llamé al [teléfono] y nada.")  # owner reply not included
    assert first.url == "https://www.google.com/maps/reviews/data=RevFakeLow0001"
    assert first.posted_at == datetime(2026, 9, 14, 15, 2, 11, tzinfo=UTC)
    assert pairs[0][1] == "Lugar en Google Maps: Contables Express SAS (Asesor contable)"
    assert pairs[3][1] == "Lugar en Google Maps: Oficina Software Nube Bogotá"
    assert "RevFakeMid0002" not in [s.external_id for s, _ in parse_places(GMAPS, "n", 2)]


def test_gmaps_review_html_becomes_plain_text():
    place = {"title": "X", "reviews": [{"reviewId": "r1", "stars": 1,
                                         "text": "Pésimo.<br><br>Soporte &amp; ventas<BR/>nada"}]}
    (signal, _), = parse_places([place], "n", 3)
    assert signal.text == "Pésimo.\n\nSoporte & ventas\nnada"


def test_tiktok_videos_skip_errors_and_scrub_captions():
    videos = parse_videos(VIDEOS)
    assert list(videos) == ["7400000000000000001", "7400000000000000002"]
    first = videos["7400000000000000001"]
    assert first.author_id == "6900000000000000001"
    assert first.caption.endswith("#contadorescolombia @usuario")
    assert video_key("https://www.tiktok.com/@a/video/7400000000000000001?lang=es") == \
        "7400000000000000001"


def test_multi_line_captions_become_one_short_line():
    caption = "Ser contador no es solo Excel.\n\n👉 Una firma mal puesta\n👉 " + "x" * 300
    video = parse_videos([{"webVideoUrl": "https://www.tiktok.com/@a/video/1", "text": caption}])
    line = video["1"].caption
    assert "\n" not in line and len(line) == 200 and line.endswith("…")
    assert line.startswith("Ser contador no es solo Excel. 👉 Una firma mal puesta 👉 xxx")


def test_tiktok_comments_skip_the_video_author():
    signals = list(parse_comments(COMMENTS, "n", parse_videos(VIDEOS)))
    assert [s.external_id for s in signals] == [
        "7400000000000100001", "7400000000000100003", "7400000000000200001"]
    assert signals[0].author_hash == hash_author("apify_tiktok", "user:7000000000000000011")
    assert signals[2].author_hash == hash_author("apify_tiktok", "comment:7400000000000200001")
    assert signals[0].posted_at == datetime(2026, 7, 2, 9, 15, tzinfo=UTC)


def test_facebook_posts_and_top_comments():
    pairs = list(parse_posts(POSTS, "n"))
    ids = [s.external_id for s, _ in pairs]
    assert ids == ["post:5550001", "comment:Y29tbWVudDo1NTUwMDAxXzU1NTAwMDI=",
                   "comment:Y29tbWVudDo1NTUwMDAxXzU1NTAwMDM=",
                   "comment:Y29tbWVudDo1NTUwMDEwXzU1NTAwMTE="]  # the text-less post is skipped
    (post, post_ctx), (comment, comment_ctx), (masked, _), (orphan, orphan_ctx) = pairs
    assert post.author_hash == hash_author("apify_facebook", "user:100000000000001")
    assert masked.author_hash == hash_author("apify_facebook", masked.external_id)
    assert post.text.endswith("Escríbanme a [correo]")
    assert post_ctx == "Grupo de Facebook: Contadores Colombia Ejemplo"
    assert comment_ctx.startswith("Comentario a esta publicación de un grupo de Facebook: ¿Alguien")
    assert "carlos@ejemplo.com" not in comment_ctx and orphan_ctx == ""
    assert comment.url.endswith("comment_id=5550002")
    only_posts = [s.external_id for s, _ in parse_posts(POSTS, "n", include_comments=False)]
    assert only_posts == ["post:5550001"]


def test_people_are_never_stored():
    signals = [s for s, _ in parse_places(GMAPS, "n", 5)]
    signals += list(parse_comments(COMMENTS, "n", parse_videos(VIDEOS)))
    signals += [s for s, _ in parse_posts(POSTS, "n")]
    contexts = [c for _, c in parse_places(GMAPS, "n", 5)] + [c for _, c in parse_posts(POSTS, "n")]
    dumped = "".join(s.model_dump_json() for s in signals) + "".join(contexts)
    dumped += "".join(v.caption for v in parse_videos(VIDEOS).values())
    for value in PEOPLE:
        assert value not in dumped
