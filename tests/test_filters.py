import pytest

from niche_miner.collectors.filters import TextFilter
from niche_miner.models import RawSignal
from niche_miner.settings import FilterSettings


def reason(text: str) -> str | None:
    return TextFilter(FilterSettings()).reason_to_drop(text)


@pytest.mark.parametrize("text", [
    "La aplicación se cierra cada vez que intento subir una factura electrónica a la DIAN.",
    "Muy lenta para registrar los gastos del mes, toca pasar todo a Excel al final.",
    "Los pedidos por WhatsApp se pierden y el cliente me escribe al 300 123 4567 bravo.",
])
def test_spanish_complaints_are_kept(text):
    assert reason(text) is None


def test_short_text_is_dropped():
    assert reason("Pésima app, no sirve para nada") == "short"


@pytest.mark.parametrize("text", [
    "The app keeps crashing every time I try to upload an invoice to the portal.",
    "O aplicativo fecha toda vez que tento enviar uma nota fiscal eletrônica para o cliente.",
])
def test_other_languages_are_dropped(text):
    assert reason(text) == "language"


@pytest.mark.parametrize("text", [
    "GANA DINERO DESDE CASA escríbeme al WhatsApp 300 123 4567 y empieza hoy mismo",
    "Visita www.ofertas-locas.com y https://promo.example.co para premios todos los días",
    "🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥 1234567890 $$$ !!!",
    "compra compra compra compra compra compra compra compra compra compra ya",
])
def test_obvious_spam_is_dropped(text):
    assert reason(text) == "spam"


def test_duplicates_within_a_run_are_dropped():
    text_filter = TextFilter(FilterSettings())
    make = lambda i, t: RawSignal(niche_tag="n", source="s", external_id=str(i), text=t)  # noqa: E731
    assert text_filter.keep(make(1, "No me deja sincronizar el inventario con la tienda virtual."))
    assert not text_filter.keep(make(2, "no me deja sincronizar el inventario con la tienda virtual"))
    assert text_filter.dropped == {"duplicate": 1}


def test_keep_strips_text_and_sets_language():
    signal = RawSignal(niche_tag="n", source="s", external_id="1",
                       text="  El soporte nunca responde y la factura sigue sin salir.  ")
    assert TextFilter(FilterSettings()).keep(signal)
    assert signal.lang == "es" and signal.text.startswith("El soporte")
