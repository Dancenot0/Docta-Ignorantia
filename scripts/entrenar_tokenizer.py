"""
Entrena el tokenizer BPE sobre nuestro propio corpus y lo guarda.

Decisión de tamaño de vocabulario: 4096.
  * Los 256 bytes base + 8 especiales ya ocupan 264.
  * Quedan ~3800 fusiones, que para un corpus tan pequeño es más que suficiente.
  * Un vocabulario más grande con poco texto daría tokens que aparecen una sola
    vez: puro desperdicio de parámetros en la capa de embedding.

Regla práctica: el tamaño del vocabulario debe ir acorde al tamaño del corpus,
no al revés. Con 6 KB de texto, 32.000 tokens de vocabulario sería absurdo.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from knowledge import BaseConocimiento                    # noqa: E402
from tokenizer import BPETokenizer, SPECIAL_TOKENS        # noqa: E402

VOCAB_SIZE = 4096


def textos_de_entrenamiento() -> list[str]:
    """Todo lo que el tokenizer va a ver: prosa + preguntas + respuestas."""
    from scripts.construir_corpus import parrafos_prosa, FUERA_DE_COBERTURA

    textos: list[str] = list(parrafos_prosa())

    base = BaseConocimiento.cargar()
    for h in base:
        textos.append(h.pregunta)
        textos.append(h.respuesta)
        textos.extend(h.hechos)

    # Los nombres de los temas también, para que "geografia" no cueste 9 tokens.
    textos.extend(h.tema.replace("/", " ") for h in base)

    # Las negativas honestas, que son una parte grande de lo que la IA dice.
    textos.extend(r for _, r in FUERA_DE_COBERTURA)

    # Frases hechas. Si no las ve, "no lo sé" se parte en bytes sueltos y gasta
    # contexto en lo que es, literalmente, su respuesta más frecuente.
    textos.extend([
        "no lo sé", "eso no lo tengo en lo que he aprendido",
        "no estoy seguro", "creo que", "prefiero decirte que no lo sé",
        "¿qué quieres saber?", "de eso no debo opinar",
        "consulta a alguien que sepa", "no voy a inventarme una respuesta",
    ])
    return textos


def main() -> None:
    textos = textos_de_entrenamiento()
    total_chars = sum(len(t) for t in textos)
    print(f"Corpus de entrenamiento del tokenizer: {len(textos)} textos, "
          f"{total_chars:,} caracteres")
    print(f"Vocabulario objetivo: {VOCAB_SIZE}\n")

    tok = BPETokenizer()
    t0 = time.time()
    tok.train(textos, vocab_size=VOCAB_SIZE, verbose=True)
    dt = time.time() - t0
    print(f"\nEntrenado en {dt:.1f}s -> vocabulario real de {len(tok)} tokens")

    ruta = RAIZ / "data" / "tokenizer.json"
    tok.save(ruta)
    print(f"Guardado en {ruta} ({ruta.stat().st_size/1024:.0f} KB)")

    # Guardamos el tamaño REAL del vocabulario (con el relleno de alineación).
    # El modelo tiene que usar exactamente este número: si la capa de embedding
    # es más pequeña que el vocabulario, revienta en el primer forward.
    meta = RAIZ / "data" / "tokenizer_meta.json"
    meta.write_text(json.dumps({
        "vocab_size": len(tok),
        "vocab_util": tok.vocab_size_real(),
        "n_merges": len(tok.merges),
        "chars_entrenamiento": total_chars,
    }, indent=2), encoding="utf-8")
    print(f"vocab_size para el modelo: {len(tok)} "
          f"({tok.vocab_size_real()} útiles) -> {meta}")

    # ------------------------------------------------------------------ #
    print("\n" + "=" * 66)
    print("PRUEBAS")
    print("=" * 66)

    casos = [
        "¿Cuál es la capital de Francia?",
        "París es la capital de Francia.",
        "El río Amazonas es el más caudaloso del mundo.",
        "no lo sé",
        "España, México, Argentina y Guinea Ecuatorial.",
        "año 2026, 3,14159, -89,2 °C",
        "emoji 🌍 y japonés 東京 y cirílico Москва",
    ]
    for texto in casos:
        ids = tok.encode(texto, add_specials=False)
        vuelta = tok.decode(ids)
        ok = "OK " if vuelta == texto else "FAIL"
        print(f"\n{ok} {texto!r}")
        print(f"     {len(ids)} tokens | {len(texto.encode('utf-8'))} bytes "
              f"| {len(texto.encode('utf-8'))/len(ids):.2f} bytes/token")
        print(f"     ida y vuelta: {vuelta!r}")

    # Idempotencia: guardar y recargar debe dar el mismo tokenizer.
    tok2 = BPETokenizer.load(ruta)
    mismos = all(tok.encode(t) == tok2.encode(t) for t in casos)
    print(f"\n{'OK ' if mismos else 'FAIL'} persistencia: "
          f"load() reproduce encode() exactamente")

    print("\n" + "=" * 66)
    print("ESTADÍSTICAS DE COMPRESIÓN")
    print("=" * 66)
    stats = tok.estadisticas(textos)
    for k, v in stats.items():
        print(f"  {k:20} {v}")

    print("\nLos 25 tokens aprendidos más útiles (por longitud):")
    largos = sorted(
        ((tid, tok.vocab[tid].decode("utf-8", "replace"))
         for tid in tok.vocab if tid >= len(SPECIAL_TOKENS) + 256),
        key=lambda kv: -len(kv[1]),
    )[:25]
    for tid, s in largos:
        print(f"  {tid:>5} {s!r}")


if __name__ == "__main__":
    main()
