#!/usr/bin/env python3
"""
Tests de `Modelo.generar()`: el bucle de muestreo autoregresivo.

Se prueban con un modelo ALEATORIO diminuto, no con el checkpoint entrenado:
lo que se verifica aquí es la mecánica del bucle (parar, suprimir, acotar,
repetir), no la calidad del texto. La calidad se comprueba en la demo con el
modelo SFT real, que es donde tiene sentido.

Ejecutar:  python tests/test_generar.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                          # noqa: E402

from config import ModelConfig                              # noqa: E402
from model import Modelo                                    # noqa: E402

# Un vocabulario y un modelo mínimos pero con las piezas reales (MoE incluido).
# Dimensiones distintas entre sí a propósito, como en check_grad: si algo se
# transpone mal, que las formas no cuadren en vez de coincidir por casualidad.
CFG = ModelConfig(
    vocab_size=41, d_model=32, n_layers=2, max_seq_len=16,
    n_heads=4, n_kv_heads=2,
    ffn_hidden=20, moe_ffn_hidden=12, shared_ffn_hidden=20,
    n_experts=4, top_k=2, shared_experts=1, dropout=0.0,
)
EOS = 1          # como en tokenizer.py: <pad>=0, <eos>=1
PROMPT = [7, 8, 9, 10]

FALLOS: list[str] = []
PASA = 0


def comprobar(nombre: str, cond: bool, detalle: str = "") -> None:
    global PASA
    if cond:
        PASA += 1
        print(f"  ✓ {nombre}")
    else:
        FALLOS.append(nombre)
        print(f"  ✗ {nombre}  {detalle}")


def main() -> int:
    m = Modelo(CFG, semilla=42)

    print("generar():")

    # 1. Voraz determinista: misma entrada, misma salida, sin azar de por medio.
    a = m.generar(PROMPT, eos=EOS, max_nuevos=6, temperatura=0.0)
    b = m.generar(PROMPT, eos=EOS, max_nuevos=6, temperatura=0.0)
    comprobar("voraz es determinista", a == b, f"{a} != {b}")

    # 2. Semilla fija con temperatura > 0 también reproduce.
    c = m.generar(PROMPT, eos=EOS, max_nuevos=6, temperatura=0.9, top_k=5,
                  semilla=7)
    d = m.generar(PROMPT, eos=EOS, max_nuevos=6, temperatura=0.9, top_k=5,
                  semilla=7)
    comprobar("misma semilla reproduce el muestreo", c == d, f"{c} != {d}")

    # 3. Si todo está suprimido menos <eos>, para inmediatamente.
    todos_menos_eos = [i for i in range(CFG.vocab_size) if i != EOS]
    e = m.generar(PROMPT, eos=EOS, max_nuevos=10, temperatura=0.8,
                  suprimir=todos_menos_eos)
    comprobar("para en cuanto sale <eos>", e == [EOS], f"salió {e}")

    # 4. Techo de max_nuevos: suprimimos TODO menos dos tokens normales —incluido
    #    <eos>—, así es imposible parar antes y el bucle tiene que agotar el
    #    presupuesto exacto.
    permitidos = {5, 6}
    suprimir = [i for i in range(CFG.vocab_size) if i not in permitidos]
    f = m.generar(PROMPT, eos=EOS, max_nuevos=5, temperatura=1.0, top_k=0,
                  suprimir=suprimir, semilla=3)
    comprobar("respeta max_nuevos exacto", len(f) == 5, f"generó {len(f)}: {f}")
    comprobar("solo genera lo permitido", set(f) <= permitidos, f"salió {set(f)}")

    # 5. Tope duro: la secuencia nunca puede pasar de max_seq_len. Prompt de
    #    13 tokens con techo 16 -> exactamente 3 nuevos y se acabó.
    g = m.generar(list(range(2, 15)), eos=EOS, max_nuevos=100,
                  temperatura=0.0, suprimir=suprimir)
    comprobar("no desborda max_seq_len", len(g) == CFG.max_seq_len - 13,
              f"generó {len(g)} con prompt de 13 y ctx {CFG.max_seq_len}")

    # 5b. Caso patológico: todo suprimido, incluido <eos>. El bucle no puede
    #     ni avanzar ni muestrear: debe parar limpio emitiendo <eos>, no
    #     sacar el argmax de una fila de -inf (que sería justo lo prohibido).
    g2 = m.generar(PROMPT, eos=EOS, max_nuevos=4, temperatura=0.8,
                   suprimir=range(CFG.vocab_size), semilla=2)
    comprobar("todo suprimido -> para limpio", g2 == [EOS], f"salió {g2}")

    # 6. El prompt del llamador no se toca (generar copia, no muta en sitio).
    original = list(PROMPT)
    m.generar(PROMPT, eos=EOS, max_nuevos=4, temperatura=0.8, semilla=1)
    comprobar("no muta el prompt", PROMPT == original)

    # 7. Lo devuelto es una lista de ints dentro del vocabulario.
    h = m.generar(PROMPT, eos=EOS, max_nuevos=4, temperatura=1.0, top_k=3,
                  semilla=5)
    comprobar("devuelve ints válidos",
              all(isinstance(t, int) and 0 <= t < CFG.vocab_size for t in h),
              f"salió {h}")

    # 8. El modelo aleatorio SIN supresiones acaba encontrando <eos> tarde o
    #    temprano... o agota el presupuesto. Lo que no puede pasar es que se
    #    cuelgue: si este test termina, el bucle termina.
    m.generar(PROMPT, eos=EOS, max_nuevos=8, temperatura=1.0, top_k=0,
              semilla=0)
    comprobar("el bucle siempre termina", True)

    print(f"\n{PASA} comprobaciones OK, {len(FALLOS)} fallos")
    if FALLOS:
        print("FALLAN:", ", ".join(FALLOS))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
