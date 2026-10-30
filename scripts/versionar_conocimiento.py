"""
Versión y manifiesto de la base de conocimiento.

Por qué existe este archivo. La arquitectura y el conocimiento cambian a ritmos
distintos y por motivos distintos. El código se toca cuando hay un bug o una
mejora de modelo; el conocimiento se toca cada vez que se aprende algo nuevo, y
eso va a ser constantemente. Tenerlos en el mismo paquete obliga a reconstruir y
volver a subir todo por añadir una ficha, y además hace imposible responder a la
única pregunta que de verdad importa cuando un modelo empieza a decir cosas
raras: ¿qué sabía exactamente cuando aprendió esto?

El manifiesto resuelve las dos cosas. Es una foto del conocimiento: qué fichas
hay, cuántas por tema, una huella de cada archivo y un número de versión que
sube solo cuando el contenido cambia de verdad.

Qué NO es. No es un control de versiones; para eso está git. Es un registro
legible por el entrenamiento, que lo usa para dos cosas concretas:

  1. saber si el corpus ya estaba visto (y por tanto si hace falta reentrenar);
  2. anotar en el checkpoint con qué conocimiento se entrenó, para que un peso
     suelto no sea un misterio.

Uso:
    python scripts/versionar_conocimiento.py            # actualiza el manifiesto
    python scripts/versionar_conocimiento.py --diff     # qué cambió desde la última
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DATOS = RAIZ / "data"
CONOCIMIENTO = DATOS / "conocimiento"
MANIFIESTO = DATOS / "MANIFIESTO.json"

# Lo que forma parte del "conocimiento" y por tanto versiona junto. El tokenizer
# está incluido a propósito: si alguien lo reentrena, el vocabulario cambia y los
# checkpoints anteriores dejan de cargar. Ese cambio tiene que quedar registrado
# junto al conocimiento, no aparte, porque rompe las dos cosas a la vez.
ARCHIVOS_CONOCIMIENTO = (
    "conocimiento/*.jsonl",
    "corpus.txt",
    "sft.jsonl",
    "tokenizer.json",
)


def huella(ruta: Path) -> str:
    """SHA-256 corto. Suficiente para saber si cambió, sin gastar espacio."""
    return hashlib.sha256(ruta.read_bytes()).hexdigest()[:16]


def escanear() -> dict:
    """Recorre los archivos de conocimiento y devuelve su estado actual."""
    archivos = {}
    for patron in ARCHIVOS_CONOCIMIENTO:
        for p in sorted(DATOS.glob(patron)):
            clave = str(p.relative_to(DATOS))
            archivos[clave] = {"bytes": p.stat().st_size, "sha256": huella(p)}

    # recuento por tema, que es lo que se quiere ver de un vistazo
    temas: dict[str, int] = {}
    n_hechos = 0
    n_con_ref = 0
    for p in sorted(CONOCIMIENTO.glob("*.jsonl")):
        for linea in p.read_text(encoding="utf-8").splitlines():
            linea = linea.strip()
            if not linea:
                continue
            d = json.loads(linea)
            n_hechos += 1
            raiz = d.get("tema", "?").split("/")[0]
            temas[raiz] = temas.get(raiz, 0) + 1
            if d.get("referencias"):
                n_con_ref += 1

    return {
        "archivos": archivos,
        "hechos": n_hechos,
        "hechos_con_referencia": n_con_ref,
        "por_tema": dict(sorted(temas.items())),
    }


def huella_global(estado: dict) -> str:
    """
    Una sola huella de todo el conocimiento.

    Es lo que permite la pregunta barata "¿cambió algo?" sin comparar archivo a
    archivo. Se calcula sobre las huellas individuales y el recuento, no sobre el
    contenido bruto, para que sea estable y rápida.
    """
    partes = [f"{k}:{v['sha256']}" for k, v in sorted(estado["archivos"].items())]
    partes.append(f"hechos:{estado['hechos']}")
    return hashlib.sha256("|".join(partes).encode()).hexdigest()[:16]


def actualizar(bump: str = "auto") -> dict:
    """
    Reescribe el manifiesto y devuelve el estado nuevo.

    La versión sube sola cuando la huella cambia. `bump` permite forzar 'mayor'
    (cambio de esquema o de tokenizer, que rompe checkpoints) o 'menor'.
    """
    estado = escanear()
    nueva_huella = huella_global(estado)

    anterior = None
    if MANIFIESTO.exists():
        anterior = json.loads(MANIFIESTO.read_text(encoding="utf-8"))

    if anterior is None:
        version = "1.0.0"
    elif anterior.get("huella") == nueva_huella and bump == "auto":
        version = anterior["version"]          # no cambió nada: misma versión
    else:
        mayor, menor, parche = (int(x) for x in anterior.get("version", "0.0.0").split("."))
        tok_cambio = (anterior.get("archivos", {}).get("tokenizer.json", {}).get("sha256")
                      != estado["archivos"].get("tokenizer.json", {}).get("sha256"))
        if bump == "mayor" or tok_cambio:
            # Cambiar el tokenizer cambia el vocabulario, y eso invalida TODOS los
            # checkpoints anteriores. No es un cambio menor por mucho que solo se
            # haya añadido una ficha: hay que decirlo fuerte.
            version = f"{mayor + 1}.0.0"
        elif bump == "menor" or estado["hechos"] > anterior.get("hechos", 0):
            version = f"{mayor}.{menor + 1}.0"
        else:
            version = f"{mayor}.{menor}.{parche + 1}"

    mani = {
        "version": version,
        "fecha": date.today().isoformat(),
        "huella": nueva_huella,
        "hechos": estado["hechos"],
        "hechos_con_referencia": estado["hechos_con_referencia"],
        "por_tema": estado["por_tema"],
        "vocabulario": _vocabulario(),
        "archivos": estado["archivos"],
        "version_anterior": anterior.get("version") if anterior else None,
        "huella_anterior": anterior.get("huella") if anterior else None,
    }
    MANIFIESTO.write_text(json.dumps(mani, indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")
    return mani


def _vocabulario() -> int:
    """Tamaño del vocabulario. Si cambia, ningún checkpoint anterior carga."""
    ruta = DATOS / "tokenizer.json"
    if not ruta.exists():
        return 0
    d = json.loads(ruta.read_text(encoding="utf-8"))
    return len(d.get("vocabulario", d.get("vocab", {})))


def cargar() -> dict | None:
    if not MANIFIESTO.exists():
        return None
    return json.loads(MANIFIESTO.read_text(encoding="utf-8"))


def diff() -> None:
    """Qué cambió desde el manifiesto guardado. Es la pregunta habitual."""
    viejo = cargar()
    nuevo = escanear()
    if viejo is None:
        print("no hay manifiesto todavía; ejecuta sin --diff para crearlo")
        return

    huella_nueva = huella_global(nuevo)
    if viejo.get("huella") == huella_nueva:
        print(f"sin cambios desde la versión {viejo['version']}")
        return

    print(f"versión registrada : {viejo['version']}")
    print(f"hechos             : {viejo['hechos']} -> {nuevo['hechos']}")
    print(f"con referencia     : {viejo['hechos_con_referencia']} -> "
          f"{nuevo['hechos_con_referencia']}")
    print(f"vocabulario        : {viejo.get('vocabulario')} -> {_vocabulario()}")

    print("\narchivos:")
    todos = sorted(set(viejo["archivos"]) | set(nuevo["archivos"]))
    for k in todos:
        a = viejo["archivos"].get(k, {}).get("sha256")
        b = nuevo["archivos"].get(k, {}).get("sha256")
        marca = "  (igual)" if a == b else ("  NUEVO" if a is None else "  CAMBIADO")
        print(f"  {k:38} {marca}")

    if viejo.get("archivos", {}).get("tokenizer.json", {}).get("sha256") != \
            nuevo["archivos"].get("tokenizer.json", {}).get("sha256"):
        print("\n⚠ EL TOKENIZER CAMBIÓ. Los checkpoints anteriores no cargan: "
              "cambia el vocabulario y con él la forma de los embeddings. "
              "Toca entrenar desde cero.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--diff", action="store_true",
                    help="solo muestra qué cambió, sin reescribir")
    ap.add_argument("--bump", choices=["auto", "menor", "mayor"], default="auto")
    a = ap.parse_args(argv)

    if a.diff:
        diff()
        return 0

    m = actualizar(a.bump)
    print(f"conocimiento v{m['version']}  ·  {m['hechos']} hechos  ·  "
          f"huella {m['huella']}")
    print(f"  con referencia documentada: {m['hechos_con_referencia']}")
    print(f"  vocabulario: {m['vocabulario']}")
    for tema, n in m["por_tema"].items():
        print(f"  {tema:22} {n:3} hechos")
    print(f"\n-> {MANIFIESTO.relative_to(RAIZ)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
