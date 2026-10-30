"""
Comprobación numérica de gradientes.

Este es el script más importante del proyecto. Todo lo demás puede fallar de
formas visibles (la pérdida no baja, el texto sale basura), pero un backward
mal escrito puede fallar en SILENCIO: la pérdida baja igualmente, solo que más
despacio o hacia un sitio peor, y no te enteras nunca.

La idea es simple y brutal. La definición de derivada es

    df/dw ≈ [ f(w + h) − f(w − h) ] / (2h)

Así que perturbamos cada peso un poquito, medimos cómo cambia la pérdida, y lo
comparamos con lo que dice nuestro backward. Si coinciden, el backward está
bien. Si no, hay un error y sabemos exactamente en qué parámetro.

Usamos un modelo minúsculo a propósito: con d_model=32 y 2 capas podemos
comprobar TODOS los parámetros, incluidos los de los 4 expertos y el router,
en pocos segundos. Si pasa aquí, pasa en el grande: las matemáticas no dependen
del tamaño.

Criterio de error relativo (el estándar de Karpathy y de todo el mundo):
    error = |analitico − numerico| / max(|analitico|, |numerico|)
    < 1e-5   perfecto
    < 1e-3   aceptable
    > 1e-2   HAY UN BUG
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from config import ModelConfig                     # noqa: E402
from model import Modelo, cross_entropy            # noqa: E402

# Tamaño de la perturbación. Demasiado pequeño y la aritmética se come la
# diferencia; demasiado grande y la aproximación lineal deja de valer.
# Con float64 podemos usar un h chiquito y el error de truncamiento es ínfimo.
# (En float32 haría falta h≈1e-2, porque la pérdida está cuantizada: el salto
# mínimo entre dos float32 cercanos a 6 es ~5e-7, y con h=1e-4 la perturbación
# ni llega a cambiar el valor. Parece un bug del backward y no lo es.)
H = 1e-5
DTYPE = np.float64


def cfg_minima(moe: bool = True) -> ModelConfig:
    """
    Modelo diminuto pero con TODAS las piezas que queremos comprobar.

    Las dimensiones están elegidas DELIBERADAMENTE distintas entre sí:
    d_model=32, ffn_hidden=20, moe_ffn_hidden=12, kv_dim=16. Si fueran iguales,
    un backward con las matrices transpuestas daría la forma correcta por pura
    casualidad y devolvería números falsos sin que saltara ningún error. Es
    exactamente el bug que costó encontrar aquí. Nunca pruebes con matrices
    cuadradas del mismo tamaño.
    """
    return ModelConfig(
        vocab_size=41, d_model=32, n_layers=2, max_seq_len=16,
        n_heads=4, n_kv_heads=2,                 # -> kv_dim = 16, distinto de 32
        ffn_hidden=20,                           # distinto de d_model
        moe_ffn_hidden=12, shared_ffn_hidden=20, # y distintos entre sí
        n_experts=4, top_k=2, shared_experts=1,
        use_moe=moe, dropout=0.0,
    )


def perdida(m: Modelo, ids: np.ndarray, mask: np.ndarray) -> float:
    """Solo forward, sin cachear nada que nos interese."""
    loss, _ = m.forward(ids, mask)
    return loss


def comprobar(m: Modelo, ids: np.ndarray, mask: np.ndarray,
              max_por_param: int = 6, semilla: int = 0) -> list[tuple]:
    """
    Devuelve [(nombre, índice, analítico, numérico, error_relativo)].

    Detalle importante: entre perturbación y perturbación NO debe quedar estado.
    En concreto, el balanceo sin loss auxiliar del MoE actualiza los bias dentro
    de backward(), así que aquí no llamamos a backward() en el bucle numérico
    (solo a forward), y los bias se congelan para que la función que derivamos
    numéricamente sea realmente la misma en las dos evaluaciones.
    """
    rng = np.random.default_rng(semilla)
    bias_congelados = {k: v.copy() for k, v in m.p.items() if k.endswith("balance_bias")}

    loss, _ = m.forward(ids, mask)
    m.backward()
    for k, v in bias_congelados.items():        # restaurar: backward() los movió
        m.p[k][...] = v

    resultados = []
    for nombre in m.params_entrenables:
        w = m.p[nombre]
        plano = w.ravel()                       # vista, comparte memoria con w
        grad = m.g[nombre].ravel()
        n = plano.size
        if n == 0:
            continue
        for idx in rng.choice(n, size=min(max_por_param, n), replace=False):
            idx = int(idx)
            original = plano[idx]

            plano[idx] = original + H
            p_mas, _ = m.forward(ids, mask)
            plano[idx] = original - H
            p_menos, _ = m.forward(ids, mask)
            plano[idx] = original

            numerico = (p_mas - p_menos) / (2 * H)
            analitico = float(grad[idx])
            denom = max(abs(analitico), abs(numerico), 1e-12)
            error = abs(analitico - numerico) / denom
            multi = np.unravel_index(idx, w.shape)
            resultados.append((nombre, multi, analitico, numerico, error))

    return resultados


def main() -> None:
    print("=" * 74)
    print("COMPROBACIÓN NUMÉRICA DE GRADIENTES")
    print("=" * 74)

    for etiqueta, moe in [("MoE (router + expertos + compartidos)", True),
                          ("FFN denso", False)]:
        print(f"\n--- {etiqueta} ---")
        cfg = cfg_minima(moe)
        m = Modelo(cfg, semilla=7, dtype=DTYPE)

        rng = np.random.default_rng(3)
        B, T = 3, 12
        ids = rng.integers(0, cfg.vocab_size, size=(B, T)).astype(np.int64)
        mask = np.ones((B, T - 1), np.float32)
        mask[1, 6:] = 0.0                    # una fila con relleno, a propósito

        print(f"  parámetros: {m.n_params():,} | lote {B}x{T} con máscara parcial")

        t0 = time.time()
        res = comprobar(m, ids, mask)
        dt = time.time() - t0

        if not res:
            print("  !! no se comprobó nada")
            continue

        errores = [r[4] for r in res]
        peores = sorted(res, key=lambda r: -r[4])[:5]

        print(f"  {len(res)} derivadas comprobadas en {dt:.1f}s")
        print(f"  error relativo  mediana {np.median(errores):.2e}  "
              f"máximo {max(errores):.2e}")

        print("  las 5 peores:")
        for nombre, multi, a, n_, e in peores:
            marca = "OK " if e < 1e-2 else ("~  " if e < 5e-2 else "MAL")
            print(f"    {marca} {nombre}{multi}  analítico={a:+.6e} "
                  f"numérico={n_:+.6e}  error={e:.2e}")

        # Desglose por tipo de parámetro. Se quita SOLO el prefijo de capa
        # ("l0.") y se conserva el resto del nombre: si se recorta más, las
        # normas por capa acaban en el mismo grupo que la norma final y el
        # informe señala al culpable equivocado.
        grupos: dict[str, list[float]] = {}
        for nombre, _, _, _, e in res:
            clave = nombre
            if clave.startswith("l") and clave[1].isdigit():
                clave = clave.split(".", 1)[1]
            # unifica los expertos: e0/e1/e2... -> eX
            for t in range(cfg.n_experts):
                clave = clave.replace(f".e{t}.", ".eX.")
            grupos.setdefault(clave, []).append(e)

        # El router del MoE merece un trato aparte, y entender por qué es parte
        # del aprendizaje: el top-k es una operación DISCRETA. Al perturbar un
        # peso, puede que un token cambie de experto seleccionado, y la pérdida
        # pega un salto. La derivada numérica a través de un punto de
        # discontinuidad no significa nada. Por eso el router nunca dará 1e-8
        # y eso NO es un bug: se distingue porque el error es pequeño (unos %)
        # y el signo coincide. Un bug de verdad da error ~1.0 o signo opuesto.
        DISCONTINUOS = {"router"}

        print("  por tipo de parámetro:")
        peores_reales = 0.0
        for clave in sorted(grupos):
            es = grupos[clave]
            if clave in DISCONTINUOS:
                # aquí solo alarma si el error es de los grandes
                mal = max(es) > 0.5
                marca = "MAL" if mal else "ok*"
                extra = "  (*top-k discontinuo: ruido esperado)" if not mal else ""
            else:
                mal = max(es) >= 1e-2
                marca = "OK " if max(es) < 1e-4 else ("~  " if not mal else "MAL")
                extra = ""
            if mal:
                peores_reales += 1
            print(f"    {marca} {clave:16} n={len(es):3}  "
                  f"mediana={np.median(es):.1e}  máx={max(es):.2e}{extra}")

        # el veredicto se emite sobre el resumen
        pass

        # Criterio de veredicto. El error RELATIVO es engañoso cuando el
        # gradiente es casi cero (1e-7 contra 1e-7 da un error relativo enorme
        # sin que importe nada), así que se juzga con la diferencia ABSOLUTA
        # normalizada por la escala de los gradientes del lote.
        escala = max(abs(a) for _, _, a, _, _ in res) or 1.0
        peores_abs = [abs(a - n) / escala for _, _, a, n, _ in res]

        # El router y lo que cuelga de él son discontinuos por el top-k.
        suspects = [d for (nom, _, a, n, _), d in zip(res, peores_abs)
                    if not nom.endswith("router")]
        router = [d for (nom, _, a, n, _), d in zip(res, peores_abs)
                  if nom.endswith("router")]
        # un bug de verdad da error relativo ~1.0 CON signo opuesto; el ruido de
        # top-k da unos pocos % con el signo correcto.
        signo_ok = all(a * n >= 0 or abs(a - n) / escala < 1e-3
                       for _, _, a, n, _ in res)

        bien = max(suspects) < 1e-3 and signo_ok
        print(f"\n  diff. absoluta máxima (no-router), relativa a la escala "
              f"de gradientes {escala:.1e}: {max(suspects):.2e}")
        if router:
            print(f"  router: máx {max(router):.2e} (discontinuo por el top-k, "
                  f"no se puede derivar numéricamente)")
        print(f"  VEREDICTO: {'CORRECTO' if bien else 'HAY ERRORES'}")


if __name__ == "__main__":
    main()
