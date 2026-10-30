"""
Bucle de entrenamiento.

forward -> backward -> recorte de gradiente -> AdamW -> repetir, con warmup y
decaimiento coseno, evaluación periódica y guardado del mejor modelo.

No tiene nada de especial, y esa es la intención: una vez que el backward está
verificado, entrenar es mecánico. Lo que sí importa es lo que se mide. Se
registran tres cosas distintas porque contestan preguntas distintas:

  pérdida de entrenamiento   ¿está optimizando? (si no baja, el bucle está roto)
  pérdida de validación      ¿está generalizando o memorizando?
  acierto del siguiente token ¿se traduce en algo observable?

Con un corpus de 2.000 tokens y un modelo de 7M parámetros la de validación va
a subir casi desde el principio. Eso no es un fallo del código: es el resultado
correcto para esta cantidad de datos, y es exactamente la señal que dice que
hace falta más texto. Verlo con números propios vale más que leerlo.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from config import get_config, estimate_params              # noqa: E402
from dataset import LotesPreentrenamiento, LotesSFT         # noqa: E402
from model import Modelo, AdamW                             # noqa: E402
from tokenizer import BPETokenizer                          # noqa: E402

DATA = RAIZ / "data"
PESOS = RAIZ / "checkpoints"


def evaluar(m: Modelo, lotes, max_lotes: int = 20) -> tuple[float, float]:
    """
    Pérdida y aciertos por token sobre un iterador de lotes.

    Ojo con las formas: `mask` ya viene como (B, T-1) desde el dataset, que es
    exactamente la forma de `pred` y de `target`. Recortarla otra vez es el
    error clásico de desalineamiento por uno, y además es silencioso: no falla,
    simplemente mide una posición menos.
    """
    total, pesos, aciertos = 0.0, 0.0, 0
    for i, (ids, mask) in enumerate(lotes):
        if i >= max_lotes:
            break
        loss, logits = m.forward(ids, mask)
        n = float(mask.sum())
        total += loss * n
        pesos += n
        pred = logits.argmax(-1)[:, :-1]        # (B, T-1): lo que predice cada posición
        target = ids[:, 1:]                     # (B, T-1): el token siguiente real
        aciertos += int(((pred == target) & (mask > 0)).sum())
    return total / max(pesos, 1.0), aciertos / max(pesos, 1.0)


def _manifiesto() -> dict | None:
    """La foto del conocimiento actual, si existe. No falla si no está."""
    ruta = DATA / "MANIFIESTO.json"
    if not ruta.exists():
        return None
    try:
        return json.loads(ruta.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def cargar_para_reanudar(ruta: str | Path, cfg, dtype=np.float32):
    """
    Carga un checkpoint para seguir entrenando, COMPROBANDO que encaja.

    No se reutiliza `Modelo.load` y esto no es manía. Aquel método hace
    `if k in datos: m.p[k] = datos[k]`, o sea: se salta en silencio las claves
    que faltan y no mira las formas. Para inferir da igual, porque el checkpoint
    y el modelo vienen del mismo sitio. Para REANUDAR no da igual: si el
    vocabulario cambió, el embedding del archivo tiene otra forma y se asigna
    tal cual, y el modelo queda con una matriz que no se corresponde con su
    propia configuración. No falla al cargar; falla tres minutos después, en el
    primer forward, con un error de dimensiones que no apunta a la causa.

    Así que aquí se exige que las claves sean exactamente las mismas y que todas
    las formas cuadren, y si no, se dice qué es lo que no cuadra.
    """
    ruta = Path(ruta)
    if not ruta.with_suffix(".npz").exists():
        raise FileNotFoundError(f"no existe el checkpoint {ruta}.npz")

    datos = np.load(ruta.with_suffix(".npz"))
    m = Modelo(cfg, dtype=dtype)

    en_archivo, en_modelo = set(datos.files), set(m.p)
    if en_archivo != en_modelo:
        raise ValueError(
            "el checkpoint no es de esta arquitectura.\n"
            f"  solo en el archivo : {sorted(en_archivo - en_modelo)[:6]}\n"
            f"  solo en el modelo  : {sorted(en_modelo - en_archivo)[:6]}\n"
            "Esto pasa al cambiar de preset o al modificar la arquitectura. "
            "Para reanudar hacen falta el mismo preset y el mismo vocabulario.")

    distintas = []
    for k in sorted(en_modelo):
        if datos[k].shape != m.p[k].shape:
            distintas.append(f"  {k}: archivo {datos[k].shape} vs modelo {m.p[k].shape}")
    if distintas:
        raise ValueError(
            "el checkpoint tiene formas incompatibles:\n" + "\n".join(distintas[:8]) +
            "\nLo habitual es que haya cambiado el vocabulario: si reentrenaste el "
            "tokenizer, los embeddings ya no miden lo mismo y no se puede continuar "
            "desde aquí. Toca entrenar desde cero.")

    for k in en_modelo:
        m.p[k] = datos[k].astype(dtype)

    meta = {}
    ruta_meta = ruta.with_suffix(".meta.json")
    if ruta_meta.exists():
        meta = json.loads(ruta_meta.read_text(encoding="utf-8"))
    return m, meta


def entrenar(modo: str, preset: str = "tiny", epocas: int = 20, batch: int = 4,
             lr: float = 3e-4, ctx: int | None = None, eval_cada: int = 5,
             guardar: bool = True, verbose: bool = True,
             reanudar: str | Path | None = None) -> dict:
    assert modo in ("pre", "sft"), modo

    tok = BPETokenizer.load(DATA / "tokenizer.json")
    cfg = get_config(preset, vocab_size=len(tok))
    if ctx:
        cfg.max_seq_len = ctx

    if modo == "pre":
        lotes = LotesPreentrenamiento(tok, DATA / "corpus.txt",
                                      ctx=min(cfg.max_seq_len, 128), solape=32)
        valid = LotesPreentrenamiento(tok, DATA / "corpus.txt",
                                      ctx=min(cfg.max_seq_len, 128), solape=0,
                                      semilla=999)
        n_pasos_epoca = max(1, len(lotes) // batch)
    else:
        lotes = LotesSFT(tok, DATA / "sft.jsonl", ctx=cfg.max_seq_len)
        valid = lotes                       # con tan pocos datos, la "validación"
        n_pasos_epoca = max(1, len(lotes) // batch)   # es el propio conjunto

    meta_anterior = {}
    if reanudar:
        m, meta_anterior = cargar_para_reanudar(reanudar, cfg)
    else:
        m = Modelo(cfg, semilla=1234)
    opt = AdamW(m, lr=lr, weight_decay=0.1)
    total_pasos = epocas * n_pasos_epoca

    if verbose:
        p = estimate_params(cfg)
        print(f"modo={modo}  preset={preset}  vocab={cfg.vocab_size}  ctx={cfg.max_seq_len}")
        if reanudar:
            # Los momentos de Adam NO se guardan, así que al reanudar el
            # optimizador arranca en frío aunque los pesos vengan calientes.
            # Conviene saberlo porque explica por qué las primeras épocas de un
            # reentrenamiento pueden empeorar la pérdida antes de mejorarla: no
            # es que se esté olvidando lo aprendido, es que Adam está volviendo a
            # estimar la escala de cada gradiente.
            print(f"reanudando desde {reanudar}")
            if meta_anterior:
                print(f"  conocimiento anterior: v{meta_anterior.get('conocimiento', '?')}"
                      f"  ·  {meta_anterior.get('hechos', '?')} hechos"
                      f"  ·  val {meta_anterior.get('mejor_val', float('nan')):.4f}")
            print("  (el optimizador arranca en frío: los momentos de Adam no se guardan)")
        print(f"parámetros: {m.n_params():,} totales | "
              f"{p['activos_por_token']:,} activos por token")
        n_desc = getattr(lotes, "descartados", 0)
        print(f"{len(lotes)} ejemplos, {n_pasos_epoca} pasos/época, "
              f"{total_pasos} pasos en total"
              + (f"  ⚠ {n_desc} descartados por no caber en ctx "
                 f"{cfg.max_seq_len}: súbelo con --ctx si quieres usarlos"
                 if n_desc else "") + "\n")

    historial = []
    paso = 0
    mejor = float("inf")
    t0 = time.time()

    for ep in range(epocas):
        m_entrenando = True
        suma, n_lotes = 0.0, 0
        for ids, mask in lotes.epoca(batch):
            opt.lr_schedule(paso, total_pasos, warmup=min(20, total_pasos // 10))
            loss, _ = m.forward(ids, mask)
            m.backward()
            norma = opt.step(grad_clip=1.0)
            suma += loss
            n_lotes += 1
            paso += 1

        if verbose and ((ep + 1) % eval_cada == 0 or ep == 0):
            l_ent = suma / max(n_lotes, 1)
            l_val, acc = evaluar(m, valid.epoca(batch))
            dt = time.time() - t0
            print(f"época {ep+1:>3}/{epocas}  ent={l_ent:.4f}  val={l_val:.4f}  "
                  f"acierto={100*acc:.1f}%  lr={opt.lr:.2e}  "
                  f"|g|={norma:.2f}  {dt:.0f}s")
            historial.append({"epoca": ep + 1, "ent": l_ent, "val": l_val,
                              "acierto": acc, "lr": opt.lr, "seg": dt})
            if l_val < mejor:
                mejor = l_val

    if guardar:
        PESOS.mkdir(parents=True, exist_ok=True)
        ruta = m.save(PESOS / f"{modo}_{preset}")
        (PESOS / f"{modo}_{preset}.hist.json").write_text(
            json.dumps(historial, indent=2), encoding="utf-8")

        # Junto a los pesos va SIEMPRE la foto del conocimiento con el que se
        # entrenaron. Un .npz suelto no dice de qué sabe, y esa es exactamente la
        # pregunta que hay que poder responder cuando el modelo empieza a
        # contestar cosas raras. Es barato y evita tener que adivinarlo.
        mani = _manifiesto()
        (PESOS / f"{modo}_{preset}.meta.json").write_text(json.dumps({
            "modo": modo,
            "preset": preset,
            "epocas": epocas,
            "lr": lr,
            "ctx": cfg.max_seq_len,
            "vocab": cfg.vocab_size,
            "mejor_val": mejor,
            "fecha": date.today().isoformat(),
            "conocimiento": (mani or {}).get("version"),
            "huella_conocimiento": (mani or {}).get("huella"),
            "hechos": (mani or {}).get("hechos"),
            "reanudado_de": str(reanudar) if reanudar else None,
            "conocimiento_anterior": meta_anterior.get("conocimiento"),
        }, indent=2, ensure_ascii=False), encoding="utf-8")

        if verbose:
            print(f"\nguardado en {ruta}.npz  (mejor val={mejor:.4f})")
            if mani:
                print(f"         conocimiento v{mani['version']} · {mani['hechos']} hechos")
            else:
                print("         ⚠ sin manifiesto: ejecuta scripts/versionar_conocimiento.py")

    if verbose and historial:
        primero, ultimo = historial[0], historial[-1]
        print(f"\nresumen: pérdida de entrenamiento {primero['ent']:.3f} -> {ultimo['ent']:.3f}")
        print(f"         pérdida de validación    {primero['val']:.3f} -> {ultimo['val']:.3f}")
        if ultimo["val"] > primero["val"]:
            print("         ⚠ la validación SUBE: el modelo está memorizando el")
            print("           corpus en vez de generalizar. Con 2.000 tokens de")
            print("           texto y millones de parámetros es lo esperado; la")
            print("           solución es más datos, no más entrenamiento.")
    return {"historial": historial, "mejor_val": mejor}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modo", choices=["pre", "sft"], default="pre")
    ap.add_argument("--preset", default="tiny",
                    help="tiny | toy | toy_dense | small | medium")
    ap.add_argument("--epocas", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ctx", type=int, default=None)
    ap.add_argument("--eval-cada", type=int, default=5)
    ap.add_argument("--no-guardar", action="store_true")
    ap.add_argument("--reanudar", default=None, metavar="RUTA",
                    help="continúa desde un checkpoint en vez de empezar de cero. "
                         "Se entrena con el corpus COMPLETO, no solo con lo nuevo, "
                         "porque entrenar solo con lo nuevo borra lo viejo.")
    a = ap.parse_args()
    entrenar(a.modo, a.preset, a.epocas, a.batch, a.lr, a.ctx,
             a.eval_cada, guardar=not a.no_guardar, reanudar=a.reanudar)
