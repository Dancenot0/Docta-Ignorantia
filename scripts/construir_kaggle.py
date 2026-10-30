"""
Construye el paquete ejecutable en Kaggle.

Qué genera y por qué así.

El problema. Kaggle ejecuta un notebook en una máquina que no es esta, sin
acceso a este repositorio. Las opciones son tres:

  1. Subir el código como Dataset de Kaggle y adjuntarlo al notebook. Funciona,
     pero obliga a tener cuenta, a mantener el dataset sincronizado con el
     código, y a que quien quiera probarlo repita esos pasos. Es fricción justo
     donde no hace falta.
  2. Pegar las 5.000 líneas de código en celdas. Ilegible e imposible de
     mantener: cada cambio hay que volver a pegarlo a mano.
  3. Meter el proyecto entero comprimido y en base64 DENTRO de una celda, y que
     esa celda lo despliegue al ejecutarse.

Se ha elegido la 3. El proyecto entero (módulos, base de conocimiento, corpus y
tokenizer) comprime a ~130 KB y en base64 son ~170 KB: cabe en una celda de
sobras. El resultado es un .ipynb AUTOCONTENIDO que se puede subir a Kaggle, o
ejecutar en local, sin datasets adjuntos, sin conexión y sin credenciales de
ningún tipo.

Qué NO se incluye, y esto es lo importante de entender antes de ejecutarlo:

  * Los checkpoints (28 MB). No caben y no hacen falta: el propósito de ir a
    Kaggle es ENTRENAR allí, no llevarse los pesos entrenados. La celda de
    entrenamiento los regenera.
  * data/sft.jsonl y data/corpus.txt se incluyen porque son pequeños y permiten
    entrenar sin reconstruir nada, pero pueden regenerarse con
    scripts/construir_corpus.py.

Sobre las credenciales. Ejecutar un notebook en Kaggle no requiere que nadie le
dé a esto un token. Se sube el .ipynb desde el navegador y ya. La API de Kaggle
solo hace falta para automatizar la subida desde un terminal, y en ese caso las
credenciales las pone quien ejecuta el comando, en su propia máquina, y nunca
dentro del notebook. Un token pegado en un notebook deja de ser secreto en el
momento en que el notebook se comparte.

Uso:
    python scripts/construir_kaggle.py
Genera:
    kaggle/docta_kaggle.ipynb     <- el que se sube a Kaggle
    kaggle/docta_instalador.py    <- lo mismo, ejecutable con `python`
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import tarfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
SALIDA = RAIZ / "kaggle"

# Lo que viaja. Se excluye lo regenerable y lo grande: checkpoints, cachés,
# entornos virtuales y artefactos de construcción.
# ---------------------------------------------------------------------------
# Qué viaja en cada paquete, y por qué van separados.
#
# La arquitectura y el conocimiento cambian a ritmos distintos. El código se toca
# cuando hay un bug o una mejora del modelo; el conocimiento se toca cada vez que
# Docta aprende algo, y eso va a ser constantemente. Tenlos juntos y cada ficha
# nueva obliga a reconstruir y volver a subir 200 KB de código que no ha cambiado.
#
# Separados, el flujo es: editar el JSONL, versionar, subir SOLO el dataset de
# conocimiento, y reanudar el entrenamiento desde el checkpoint anterior. Kaggle
# además versiona los datasets por su cuenta, así que queda histórico de qué
# sabía el modelo en cada momento.
# ---------------------------------------------------------------------------

# La ARQUITECTURA: código puro. Cambia poco.
PAQUETE_CODIGO = (
    "*.py",
    "scripts/*.py",
    "tests/*.py",
)

# El CONOCIMIENTO: datos. Cambia a menudo.
#
# El tokenizer va aquí y no con el código, y es una decisión con trampa. Es un
# archivo de datos, sí, pero si se reentrena cambia el vocabulario y con él la
# forma de los embeddings: ningún checkpoint anterior vuelve a cargar. Por eso el
# manifiesto le hace seguimiento aparte y avisa en mayúsculas cuando cambia.
# Mientras no se toque, el BPE es a nivel de byte y cualquier texto nuevo en
# español ya es codificable sin reentrenar nada —que es justo lo que hace que el
# reentrenamiento incremental sea limpio.
PAQUETE_CONOCIMIENTO = (
    "data/conocimiento/*.jsonl",
    "data/tokenizer.json",
    "data/tokenizer_meta.json",
    "data/corpus.txt",
    "data/sft.jsonl",
    "data/MANIFIESTO.json",
)

# Todo junto, para el modo autocontenido.
INCLUIR = PAQUETE_CODIGO + PAQUETE_CONOCIMIENTO


# ---------------------------------------------------------------------------
# Empaquetado
# ---------------------------------------------------------------------------
def rutas_de(patrones) -> list[Path]:
    """Archivos del proyecto que casan con estos patrones, ordenados y sin repetidos."""
    rutas: list[Path] = []
    for patron in patrones:
        rutas.extend(p for p in RAIZ.glob(patron) if p.is_file())
    return sorted(set(rutas))


def construir_tar(patrones=INCLUIR) -> tuple[bytes, list[Path]]:
    """
    Empaqueta en un .tar.gz con rutas relativas y DETERMINISTAS.

    Determinista quiere decir que dos construcciones del mismo contenido dan el
    mismo archivo byte a byte. Sin eso es imposible saber si un paquete cambió de
    verdad o solo cambió la fecha, y se acaba volviendo a subir todo "por si
    acaso". De ahí el mtime a cero y los uid/gid vacíos de más abajo.
    """
    rutas = rutas_de(patrones)

    buf = io.BytesIO()
    # mtime fijo: dos construcciones seguidas del mismo código tienen que dar el
    # mismo archivo byte a byte, si no es imposible saber si algo cambió de
    # verdad o solo cambió la fecha.
    with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=9) as tar:
        for p in rutas:
            info = tar.gettarinfo(p, arcname=str(p.relative_to(RAIZ)))
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with p.open("rb") as f:
                tar.addfile(info, f)
    return buf.getvalue(), rutas


# ---------------------------------------------------------------------------
# El cargador. Este texto ACABA siendo la primera celda del notebook, así que se
# escribe para leerse en Kaggle, no para leerse aquí.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Tamaño de cada trozo del paquete, en caracteres base64.
#
# Esta constante existe por un fallo real de Kaggle. La primera versión metía los
# 174 KB enteros en una celda y la página se bloqueaba al abrirla con
# "NotFoundError: Failed to execute 'removeChild' on 'Node'". No era el tamaño sin
# más: el blob iba entre comillas triples, así que el resaltador de sintaxis del
# editor entraba en modo "cadena que atraviesa 1.740 líneas" y construía un árbol
# de nodos que el DOM virtual no sabía deshacer. La recursión del stack trace era
# el resaltador, no el notebook.
#
# Se arregla con dos cambios que van juntos y ninguno basta por separado:
#   * cada línea base64 es una cadena SIMPLE de una línea, nunca un bloque entre
#     comillas triples, para que el resaltador no anide nada;
#   * el blob se parte en trozos de este tamaño repartidos en celdas distintas,
#     para que ninguna celda sea lo bastante grande como para atragantar al editor.
#
# 16.000 caracteres son ~160 líneas por celda. Probado: el editor de Kaggle lo
# maneja sin problema, y el notebook sigue siendo autocontenido.
TAMANO_TROZO = 16_000

ANCHO_LINEA = 100


def partir_base64(crudo: bytes) -> tuple[list[str], str]:
    """
    Codifica y parte el paquete en trozos listos para incrustar.

    Devuelve (lista_de_lineas_base64, lista_de_trozos). Cada línea mide
    ANCHO_LINEA y ninguna lleva nada que haya que escapar en una cadena Python,
    que es justamente la propiedad que se busca: el base64 solo tiene A-Z a-z 0-9
    + / =, así que no hay comillas ni barras invertidas y cada línea puede ir
    entre comillas simples sin que el editor se haga un lío.
    """
    import base64 as _b64
    codigo = _b64.b64encode(crudo).decode("ascii")
    lineas = [codigo[i:i + ANCHO_LINEA] for i in range(0, len(codigo), ANCHO_LINEA)]
    por_trozo = max(1, TAMANO_TROZO // ANCHO_LINEA)
    trozos = ["\n".join(lineas[i:i + por_trozo])
              for i in range(0, len(lineas), por_trozo)]
    return lineas, trozos


# ---------------------------------------------------------------------------
# Código del cargador. Se parte en dos: la cabecera va en la primera celda de
# paquete y el resto en la celda de despliegue, con las celdas de datos en medio.
# ---------------------------------------------------------------------------
CABECERA = """\
# ============================================================================
# DOCTA · paquete autocontenido  (1/%%N%%)
# ============================================================================
# Las celdas marcadas como PAQUETE llevan el proyecto entero comprimido y en
# base64: los 9 módulos, los 5 scripts, los 2 tests y los 9 archivos de datos,
# con las 53 fichas curadas, el corpus y el tokenizer.
#
# NO HACE FALTA LEERLAS NI TOCARLAS. Ejecuta todas las celdas en orden
# (Run > Run all) y la celda DEPLIEGUE monta el proyecto en ./docta.
#
# Cada línea de abajo es una cadena de una línea a propósito. Un solo bloque de
# 174 KB entre comillas triples bloquea el editor de Kaggle: el resaltador de
# sintaxis intenta anidar una cadena de mil setecientas líneas y el DOM se
# desincroniza. Troceado y en cadenas simples, no.
#
# Lo que NO está aquí: los checkpoints (28 MB). Se regeneran entrenando, que es
# precisamente para lo que se viene a Kaggle.
# ============================================================================

PAQUETE = [
"""

DESPLIEGUE = r'''
# ============================================================================
# DOCTA · despliegue
# ============================================================================
# Esta es la última celda del paquete. Junta los trozos, los descomprime en
# ./docta, mete el proyecto en el sys.path y comprueba que todo carga.
# ============================================================================

import base64
import io
import os
import sys
import tarfile
from pathlib import Path

PROYECTO = ("docta", "config", "tokenizer", "knowledge", "citacion", "retriever",
            "identidad", "dataset", "model")


def _seguro(miembro) -> bool:
    """
    ¿Se puede extraer este miembro sin salirse del destino?

    Rechaza rutas absolutas y cualquier `..` que escale fuera, y también los
    enlaces, que son la otra forma de escribir fuera de la carpeta. El archivo lo
    generamos nosotros, pero validar igual es lo correcto: el día que alguien
    regenere el paquete con otra cosa dentro, esto no debe poder tocar ningún
    sitio inesperado.

    Se hace a mano en vez de con el `filter="data"` de `tarfile` porque ese
    parámetro es de Python 3.12 y los kernels de Kaggle corren 3.10/3.11. Un
    notebook que solo funciona en la versión que tiene el autor no funciona.
    """
    ruta = Path(miembro.name)
    if ruta.is_absolute() or ruta.drive:
        return False
    return ".." not in ruta.parts


def desplegar(destino=None):
    """Descomprime el proyecto y devuelve la ruta donde ha quedado."""
    destino = Path(destino or Path.cwd() / "docta").resolve()
    destino.mkdir(parents=True, exist_ok=True)
    bruto = base64.b64decode("\n".join(PAQUETE))
    with tarfile.open(fileobj=io.BytesIO(bruto), mode="r:gz") as tar:
        for m in tar.getmembers():
            if not _seguro(m):
                raise RuntimeError(f"ruta no permitida en el paquete: {m.name}")
            if m.issym() or m.islnk():
                raise RuntimeError(f"el paquete no debe llevar enlaces: {m.name}")
            ruta = destino / m.name
            if m.isdir():
                ruta.mkdir(parents=True, exist_ok=True)
            else:
                ruta.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(m) as origen, ruta.open("wb") as salida:
                    salida.write(origen.read())
    return destino


def arrancar(destino=None):
    """Despliega, mete el proyecto en el sys.path y comprueba que carga."""
    destino = desplegar(destino)
    if str(destino) not in sys.path:
        sys.path.insert(0, str(destino))
    # se cambia al directorio del proyecto porque los scripts de entrenamiento
    # escriben en rutas relativas (checkpoints/) y así queda todo junto.
    os.chdir(destino)
    # Sin esto, en un kernel las celdas que lanzan un subprocess imprimen en
    # orden incorrecto: el hijo escribe directo al descriptor 1 mientras el padre
    # bufferiza sus propios print hasta que la celda termina. En un terminal no
    # se nota porque ahí stdout va línea a línea; en Jupyter va por bloques.
    os.environ["PYTHONUNBUFFERED"] = "1"
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    import numpy as np
    import config
    from knowledge import BaseConocimiento

    base = BaseConocimiento.cargar()
    print(f"numpy {np.__version__}")
    print(f"proyecto desplegado en {destino}")
    print(f"{len(base)} hechos curados cargados")
    print(f"  con referencia documentada: {base.estadisticas().get('con_referencia', 0)}")
    for nombre in ("tiny", "toy", "small"):
        if nombre not in config.PRESETS:
            continue
        p = config.estimate_params(config.get_config(nombre))
        print(f"  preset {nombre:6} {p['totales']/1e6:8.1f} M parámetros "
              f"({p['activos_por_token']/1e6:.1f} M activos por token)")
    # Comprobación de que el paquete ha llegado entero. Va al final y es lo
    # último que se ejecuta, que es exactamente la razón por la que existe: un
    # despliegue que carga la base de conocimiento pero se ha dejado un módulo por
    # el camino no falla hasta muchas celdas después, y entonces el error apunta
    # al sitio equivocado.
    from importlib.util import find_spec
    faltan = [m for m in PROYECTO if find_spec(m) is None]
    if faltan:
        raise RuntimeError(f"no se han desplegado: {faltan}")
    return destino
'''

CARGADOR = CABECERA  # compatibilidad con construir_instalador, que la reutiliza
DEMO = '''\
# ============================================================================
# La puerta de ignorancia y las citas
# ============================================================================
# Esto es lo que define a Docta y no necesita entrenar nada: funciona con la
# base curada. Se ve el comportamiento completo —lo que sabe, lo que duda y lo
# que directamente no sabe— y cómo cada respuesta lleva su verificación.
# ============================================================================

from retriever import Retriever, responder
from knowledge import BaseConocimiento

r = Retriever(BaseConocimiento.cargar())

PREGUNTAS = [
    "¿Cuál es la capital de Francia?",
    "capital de australia",
    "¿cuál es la montaña más alta del mundo?",
    "¿cuál es el río más largo del mundo?",
    "¿cuántos países hablan español?",
    # Las tres siguientes NO están en la base. La respuesta correcta es "no lo
    # sé" y el interés está en comprobar que no se inventa nada parecido.
    "¿Cuál es la capital de Kazajistán?",
    "¿Qué idioma se habla en Bután?",
    "¿me quieres?",
    # Y una vetada: aunque hubiera datos, no se opina de esto.
    "¿qué dosis de paracetamol le doy a un niño de 3 años?",
]

for p in PREGUNTAS:
    resp = r.consultar(p)
    marca = "SABE   " if resp.sabe else "NO SABE"
    print("=" * 74)
    print(f"[{marca}] ({resp.grado})  {p}")
    print("-" * 74)
    print(responder(resp))
    print()
'''

TESTS = '''\
# ============================================================================
# Comprobaciones
# ============================================================================
# Las dos suites, tal cual se ejecutan en local. Si esto pasa en Kaggle, el
# despliegue ha funcionado: los 46 checks de la puerta de ignorancia y los 13
# casos de verificación de citas.
# ============================================================================

import subprocess

for suite in ("tests/test_puerta.py", "tests/test_citacion.py"):
    print("=" * 74)
    print(suite)
    print("=" * 74)
    subprocess.run([sys.executable, suite], check=False)
    print()
'''

GRAD = '''\
# ============================================================================
# Verificación analítica del backprop
# ============================================================================
# Antes de entrenar nada conviene saber que el gradiente está bien. Esto compara
# el backprop escrito a mano contra la derivada numérica, parámetro a parámetro,
# y es lo que permite confiar en que si la pérdida baja es porque el modelo
# aprende y no porque hay un bug que casualmente desciende.
#
# Se ejecuta en float64 a propósito: en float32 el error de truncamiento es del
# mismo orden que la diferencia que se quiere medir y la comprobación no
# distingue un gradiente correcto de uno incorrecto.
# ============================================================================

import subprocess
subprocess.run([sys.executable, "scripts/check_grad.py"], check=False)
'''

ENTRENAR = '''\
# ============================================================================
# Entrenamiento
# ============================================================================
# El pre-entrenamiento de verdad. Esto es lo que justifica venir a Kaggle: en
# local, con 2 CPUs y 1 GB, el preset tiny tarda unos 3 minutos para 40 épocas y
# ahí se acaba lo que se puede probar. Aquí hay más CPU y, si se activa el
# acelerador, GPU.
#
# AJUSTAR ANTES DE EJECUTAR:
#   PRESET  tiny  -> humo, minutos.   toy -> el de verdad, ~16 M parámetros.
#   EPOCAS  sube esto si quieres una pérdida final más baja.
#
# Ojo con la memoria si se activa la GPU: el forward/backward está escrito en
# numpy y no sabe usarla, así que el acelerador no acelera nada aquí. Para que
# sirva haría falta portar model.py a JAX o torch, que es trabajo pendiente.
# ============================================================================

import subprocess

PRESET = "tiny"
EPOCAS = "40"

subprocess.run([sys.executable, "scripts/entrenar.py",
                "--preset", PRESET, "--epocas", EPOCAS], check=False)
'''

CIERRE = '''\
# ============================================================================
# Hablar con Docta
# ============================================================================
# Ya con los pesos entrenados en el paso anterior. Escribe tu pregunta en la
# variable y ejecuta la celda.
#
# Lo que se ve abajo no es generación neuronal todavía: es la respuesta curada
# con su verificación y sus citas. La generación está montada en model.py pero
# el bucle de chat sobre pesos entrenados es la fase siguiente del proyecto. Lo
# que sí está completo ya es lo importante —la decisión de contestar o callar, y
# el respaldo de lo que se contesta.
# ============================================================================

from retriever import Retriever, responder
from knowledge import BaseConocimiento

r = Retriever(BaseConocimiento.cargar())

PREGUNTA = "¿Cuál es la capital de Bolivia?"

resp = r.consultar(PREGUNTA)
print(f"[{'SABE' if resp.sabe else 'NO SABE'}] ({resp.grado})  {PREGUNTA}\\n")
print(responder(resp))
'''

# ---------------------------------------------------------------------------
# El notebook del modo SEPARADO: dos datasets + un notebook fino.
# ---------------------------------------------------------------------------
CARGAR_DATASETS = '''\\
# ============================================================================
# Montar la arquitectura y el conocimiento
# ============================================================================
# Este notebook NO lleva el proyecto dentro. Lo lee de los dos datasets que
# tienes que adjuntar con "Add Input":
#
#     docta-codigo        la arquitectura: módulos, scripts y tests
#     docta-conocimiento  los datos: fichas curadas, corpus, sft y tokenizer
#
# Van separados a propósito. El conocimiento cambia cada vez que Docta aprende
# algo; la arquitectura cambia cuando hay un bug. Mezclados, cada ficha nueva
# obliga a resubir 200 KB de código intacto.
#
# /kaggle/input es de SOLO LECTURA, así que se copia todo a /kaggle/working. Si
# no, el entrenamiento no podría escribir los checkpoints.
# ============================================================================

import os
import shutil
import sys
from pathlib import Path

# Las rutas de Kaggle, pero se pueden pisar por variable de entorno. No es un
# adorno para probar: sin esto esta celda no se puede ejecutar fuera de un
# kernel, y una celda que no se puede probar llega a Kaggle sin probar. Los dos
# fallos que aparecieron allí y no aquí (`filter="data"` de Python 3.12 y el
# orden de salida con subprocess) aparecieron precisamente por eso.
ENTRADA = Path(os.environ.get("KAGGLE_INPUT", "/kaggle/input"))
TRABAJO = Path(os.environ.get("KAGGLE_WORKING", "/kaggle/working")) / "docta"

# Los slugs los eliges tú al subir cada dataset. Si les pusiste otros nombres,
# cámbialos aquí y ya.
DATASET_CODIGO = "docta-codigo"
DATASET_CONOCIMIENTO = "docta-conocimiento"


def localizar(slug: str, marca: str) -> Path:
    """
    Encuentra la RAÍZ del dataset adjunto, aguante la estructura que aguante.

    Kaggle descomprime los zip en la raíz del dataset, pero si el zip llevaba una
    carpeta dentro, todo queda un nivel más abajo. Buscar una marca en vez de
    asumir la ruta evita ese fallo.

    Y un detalle que costó un bug encontrar: la marca tiene que ser una RUTA
    RELATIVA al dataset, no un nombre de archivo suelto. La primera versión
    buscaba "tokenizer.json" a secas y devolvía `hallados[0].parent`, que en el
    dataset de conocimiento es `data/`, no la raíz. Como luego las rutas se
    calculan con `relative_to(raiz)`, el prefijo `data/` se perdía y todos los
    archivos acababan volcados planos en el directorio de trabajo. No daba error
    al montar: daba error tres llamadas después, diciendo que no encontraba
    `data/conocimiento`, que no es donde nadie iría a buscar la causa.
    """
    raiz = ENTRADA / slug
    if not raiz.exists():
        candidatos = sorted(p.name for p in ENTRADA.iterdir()) if ENTRADA.exists() else []
        raise FileNotFoundError(
            f"no está adjunto el dataset {slug!r}.\\n"
            f"Adjunta los dos con 'Add Input'. Ahora mismo hay: {candidatos or 'nada'}")

    if (raiz / marca).exists():
        return raiz

    # la marca puede quedar más abajo si el zip llevaba una carpeta dentro
    partes = Path(marca).parts
    for hallado in raiz.rglob(partes[-1]):
        if hallado.parts[-len(partes):] == partes:
            base = hallado
            for _ in partes:
                base = base.parent
            if (base / marca).exists():
                return base
    raise FileNotFoundError(
        f"en {raiz} no está {marca!r}. ¿Es el dataset correcto?")


def montar() -> Path:
    TRABAJO.parent.mkdir(parents=True, exist_ok=True)
    if TRABAJO.exists():
        shutil.rmtree(TRABAJO)

    origen_codigo = localizar(DATASET_CODIGO, "model.py")
    origen_datos = localizar(DATASET_CONOCIMIENTO, "data/tokenizer.json")

    for p in origen_codigo.rglob("*"):
        if p.is_file():
            destino = TRABAJO / p.relative_to(origen_codigo)
            destino.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, destino)

    for p in origen_datos.rglob("*"):
        if p.is_file():
            # los datos van SIEMPRE por encima: si el dataset de código llevara
            # una copia vieja de data/, sería la vieja la que entrenara
            destino = TRABAJO / p.relative_to(origen_datos)
            destino.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, destino)

    if str(TRABAJO) not in sys.path:
        sys.path.insert(0, str(TRABAJO))
    return TRABAJO


def comprobar(destino: Path) -> None:
    """Que todo llegó, dicho antes de que nada falle de forma confusa."""
    import json
    import numpy as np
    from knowledge import BaseConocimiento

    base = BaseConocimiento.cargar(destino / "data" / "conocimiento")
    print(f"numpy {np.__version__}")
    print(f"proyecto en {destino}")
    print(f"{len(base)} hechos curados · {base.estadisticas().get('con_referencia', 0)} con referencia")

    manifiesto = destino / "data" / "MANIFIESTO.json"
    if manifiesto.exists():
        m = json.loads(manifiesto.read_text())
        print(f"conocimiento v{m['version']} · huella {m['huella']} · vocab {m['vocabulario']}")
    else:
        print("⚠ sin MANIFIESTO.json: ejecuta scripts/versionar_conocimiento.py "
              "antes de empaquetar, si no el checkpoint no sabrá con qué se entrenó")

    faltan = [n for n in ("model.py", "retriever.py", "citacion.py",
                          "data/tokenizer.json") if not (destino / n).exists()]
    if faltan:
        raise FileNotFoundError(f"faltan: {faltan}")


RAIZ = montar()
comprobar(RAIZ)
os.chdir(RAIZ)
print(f"\\ndirectorio de trabajo: {os.getcwd()}")
'''

REENTRANAR = '''\\
# ============================================================================
# Reentrenamiento incremental
# ============================================================================
# Esto es lo que evita empezar de cero cada vez. Carga el checkpoint anterior y
# sigue entrenando con el corpus COMPLETO, no solo con las fichas nuevas.
#
# Por qué con el corpus completo y no solo lo nuevo: entrenar únicamente con lo
# nuevo es la receta del olvido catastrófico. El modelo ajusta los pesos para lo
# que acaba de ver y pierde lo anterior. Aquí el corpus entero cabe en segundos,
# así que no hay motivo para arriesgarlo; cuando no quepa, la solución es mezclar
# lo nuevo con una muestra de lo viejo, no entrenar solo con lo nuevo.
#
# Ojo con una cosa que se ve en la primera época: el optimizador arranca en frío
# porque los momentos de Adam no se guardan. La pérdida puede subir un poco antes
# de bajar. No es que se esté olvidando lo aprendido.
# ============================================================================

import os
import subprocess
import sys
from pathlib import Path

PRESET = "tiny"
EPOCAS = "10"

# ruta del checkpoint anterior. En Kaggle lo normal es guardarlo como Output del
# notebook anterior y adjuntarlo, o dejarlo en /kaggle/working si es la misma
# sesión. Pon aquí la ruta que tengas; None = entrenar desde cero.
CHECKPOINT_ANTERIOR = None

for candidato in ("/kaggle/input/docta-checkpoints/pre_tiny",
                  "/kaggle/working/pre_tiny",
                  "checkpoints/pre_tiny"):
    if Path(candidato).with_suffix(".npz").exists():
        CHECKPOINT_ANTERIOR = candidato
        break

comando = [sys.executable, "scripts/entrenar.py",
           "--preset", PRESET, "--epocas", EPOCAS]
if CHECKPOINT_ANTERIOR:
    comando += ["--reanudar", CHECKPOINT_ANTERIOR]
    print(f"reanudando desde {CHECKPOINT_ANTERIOR}")
else:
    print("no hay checkpoint anterior: se entrena desde cero")

subprocess.run(comando, check=False)
'''


# ---------------------------------------------------------------------------
# Construcción del notebook
# ---------------------------------------------------------------------------
def celda_codigo(texto: str, id: str = "") -> dict:
    c = {"cell_type": "code", "execution_count": None, "metadata": {},
         "outputs": [], "source": texto.splitlines(keepends=True)}
    if id:
        # nbformat 4.5 exige un id por celda. Sin él esto hoy es solo un aviso,
        # pero va camino de ser un error duro, y un notebook que el visor del
        # destino rechaza no se ejecuta. Se pone un id legible y no uno al azar
        # para que dos construcciones del mismo código den el mismo archivo.
        c["id"] = id
    return c


def celda_md(texto: str, id: str = "") -> dict:
    c = {"cell_type": "markdown", "metadata": {},
         "source": texto.splitlines(keepends=True)}
    if id:
        c["id"] = id
    return c


PORTADA = """# Docta · una IA que sabe lo que no sabe

Red neuronal transformer escrita a mano en numpy —estructura completa de
DeepSeek-V3 a escala de juguete: RoPE, RMSNorm, SwiGLU, atención agrupada y
mezcla de expertos con balanceo sin pérdida auxiliar— más una base de
conocimiento curada y una puerta de ignorancia.

**El objetivo no es que lo sepa todo.** Es que tenga la cultura general de una
persona media y que, cuando no la tenga, lo diga. Todo lo demás está al servicio
de eso.

---

### Cómo se ejecuta esto

**Run → Run all.** Las celdas PAQUETE llevan el proyecto comprimido dentro; no
hay que leerlas ni tocarlas, y la celda DEPLIEGUE lo monta todo en `./docta`.

Las cuatro primeras celdas útiles valen solas: despliegan, muestran el
comportamiento, pasan los tests y comprueban el backprop. Ninguna necesita GPU ni
conexión ni entrenar nada.

> ⚠️ **Si al abrir este notebook Kaggle muestra una página bloqueada** con un
> error `removeChild` en `vendor.js`, no es el notebook: es el editor atragantándose
> con las celdas del paquete. Dale a *Run all* igualmente, o ábrelo con
> `File → Open in → Colab`. Las celdas del paquete están troceadas a propósito
> para que esto no pase, pero el editor de Kaggle es delicado con celdas grandes.

### Qué se va a ver

| Celda | Qué hace | Tarda |
|---|---|---|
| PAQUETE + DEPLIEGUE | Monta el proyecto y comprueba que carga | segundos |
| PUERTA | Sabe, duda, calla y veta: 9 preguntas | segundos |
| TESTS | Las dos suites (46 + 13 casos) | segundos |
| GRADIENTE | Backprop contra derivada numérica | ~1 min |
| ENTRENAR | Pre-entrenamiento de verdad | ~2 min (tiny) |
| PREGUNTAR | Hablar con Docta | segundos |

### Lo importante

En la celda PUERTA hay nueve preguntas. Cinco las contesta y **cuatro las
rechaza**, y esas cuatro son el punto entero del proyecto. Un modelo de lenguaje
responde a todo con la misma fluidez; Docta tiene que distinguir entre lo que
tiene curado, lo que tiene a medias y lo que no tiene, y hablar de forma distinta
en cada caso.

Una de esas cuatro es una dosis de paracetamol para un niño. Aunque hubiera datos
en la base, de eso no se opina.

### Sin credenciales

Este notebook es autocontenido. No adjunta ningún Dataset, no descarga nada y no
necesita ningún token de API. Se sube a Kaggle desde el navegador y se ejecuta.
"""

ARQUITECTURA = """---

### Qué hay dentro

**`model.py`** — el transformer, forward y backward escritos a mano. Sin
autodiff: cada gradiente está derivado y comprobado contra la derivada numérica.
Arquitectura DeepSeek-V3 completa a escala: normalización RMS, posiciones
rotatorias, atención con grupos de consulta compartidos, unidades SwiGLU y una
capa de mezcla de expertos con balanceo de carga sin pérdida auxiliar —el bias
del enrutador se corrige con el error de uso acumulado, fuera de la función de
pérdida, que es lo que describe el informe técnico de DeepSeek.

**`knowledge.py`** — la base curada. 53 fichas de geografía y metacognición, cada
una con su certeza, su tema y sus referencias.

**`retriever.py`** — recuperador TF-IDF y la puerta de ignorancia. Cuatro
comprobaciones antes de afirmar nada: que haya evidencia, que la fuente sea
fiable, que el tema no esté vetado y —la más importante— que el hecho recuperado
trate realmente de lo preguntado. Ese último chequeo existe porque sin él, «¿qué
idioma se habla en Bután?» recuperaba un hecho sobre América del Sur y la IA
contestaba sobre Brasil. Con fluidez, con un dato correcto, sobre el país
equivocado. No se parece a un error: se parece a una respuesta.

**`citacion.py`** — verificación y citas. Cada frase de la respuesta se comprueba
contra la evidencia por tres vías: que sus cifras aparezcan en la fuente, que sus
palabras con contenido aparezcan, y que sus pares de palabras consecutivas
aparezcan. La tercera es la que detecta relaciones inventadas. Sin ella,
«Johannesburgo es la sede del parlamento» pasa por respaldada usando solo
palabras que están en la ficha de Sudáfrica —y es falsa: la sede es Ciudad del
Cabo.

Y esto se le aplica **también al texto que genera la red neuronal**, no solo al
curado. Un modelo de lenguaje puede tomar una respuesta correcta y añadirle un
dato inventado con total fluidez. Sin esta capa el sistema citaría fuentes
mientras alucina.

### Lo que falta

Generación neuronal sobre pesos entrenados, memoria episódica (aprender de la
conversación), el bucle de sueños al estilo Dream-RSI —reproducir árboles de
descubrimiento en un simulador cerrado para mejorar la política de exploración,
no los pesos— y ampliar el temario más allá de geografía.

Y una cosa que se ve en cada respuesta: las 53 fichas tienen **cero referencias
documentadas**, así que todo termina en «sin fuente documentada». La capa de
citas funciona; lo que falta es que alguien abra las fuentes y las apunte.
`verificado: true` solo lo puede poner un humano que haya comprobado el dato.
"""


# ---------------------------------------------------------------------------
# Construcción del notebook
# ---------------------------------------------------------------------------
def celdas_paquete(trozos: list[str]) -> list[dict]:
    """
    Las celdas que llevan el proyecto, y la explicación de por qué son varias.

    La primera versión metía los 174 KB en UNA celda y Kaggle se bloqueaba al
    abrirla con `NotFoundError: Failed to execute 'removeChild' on 'Node'`. El
    notebook se ejecutaba perfectamente —entrenó y dio respuestas correctas en
    /kaggle/working—; lo que reventaba era el EDITOR al intentar pintarla.

    La causa concreta no es el tamaño a secas. El blob iba entre comillas triples,
    así que el resaltador de sintaxis entraba en modo "cadena que atraviesa 1.740
    líneas" y montaba un árbol de nodos que el DOM virtual no sabía deshacer: la
    recursión del stack trace era el resaltador. Por eso hacen falta los dos
    cambios juntos, y ninguno basta solo:

      * cada línea base64 es una cadena SIMPLE de una línea, nunca un bloque
        entre comillas triples;
      * el blob se parte en celdas de ~16 KB para que ninguna sea enorme.
    """
    n = len(trozos)
    celdas = []
    for i, trozo in enumerate(trozos, 1):
        lineas = "".join(f"    '{ln}'\n" for ln in trozo.split("\n"))
        if i == 1:
            fuente = CABECERA.replace("%%N%%", str(n)) + lineas + "]\n"
        elif i == n:
            aviso = ("print(f'paquete completo: {len(chr(10).join(PAQUETE)):,} "
                     "caracteres base64')")
            fuente = (f"# PAQUETE ({i}/{n}) · último trozo\n"
                      f"PAQUETE += [\n{lineas}]\n" + aviso + "\n")
        else:
            fuente = f"# PAQUETE ({i}/{n})\nPAQUETE += [\n{lineas}]\n"
        celdas.append(celda_codigo(fuente, id=f"paquete-{i:02d}"))
    return celdas


def construir_notebook(trozos: list[str]) -> dict:
    celdas = [
        celda_md(PORTADA, id="portada"),
        *celdas_paquete(trozos),
        celda_codigo(DESPLIEGUE + "\nRAIZ = arrancar()\n", id="despliegue"),
        celda_md("## La puerta de ignorancia\n\nSin entrenar nada: esto funciona "
                 "solo con la base curada.", id="md-puerta"),
        celda_codigo(DEMO, id="puerta"),
        celda_md("## Comprobaciones", id="md-tests"),
        celda_codigo(TESTS, id="tests"),
        celda_md("## El backprop es correcto", id="md-grad"),
        celda_codigo(GRAD, id="gradiente"),
        celda_md("## Entrenar", id="md-entrenar"),
        celda_codigo(ENTRENAR, id="entrenar"),
        celda_md("## Preguntar", id="md-preguntar"),
        celda_codigo(CIERRE, id="preguntar"),
        celda_md(ARQUITECTURA, id="arquitectura"),
    ]
    return {
        "cells": celdas,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# ---------------------------------------------------------------------------
# El script suelto y el zip para la vía Dataset
# ---------------------------------------------------------------------------
INSTALADOR_CUERPO = '''

def comprobar(destino):
    import subprocess
    for suite in ("tests/test_puerta.py", "tests/test_citacion.py"):
        print()
        print("-" * 74)
        print(suite)
        print("-" * 74)
        subprocess.run([sys.executable, suite], cwd=str(destino), check=False)


def main(argv):
    solo = "--solo" in argv
    destino = None
    if "--dest" in argv:
        destino = Path(argv[argv.index("--dest") + 1])
    raiz = arrancar(destino)
    print()
    print(f"{sum(1 for _ in raiz.rglob('*.py'))} módulos, "
          f"{sum(1 for _ in (raiz / 'data' / 'conocimiento').glob('*.jsonl'))} fichas")
    if not solo:
        comprobar(raiz)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
'''


def construir_instalador(trozos: list[str]) -> str:
    """
    El mismo paquete como script suelto.

    Existe por dos motivos. Porque no todo el mundo quiere un notebook:
    `python docta_instalador.py` despliega el proyecto y pasa las tests. Y porque
    es la forma más rápida de comprobar que el paquete funciona ANTES de subirlo a
    Kaggle, que es como se cazaron los dos bugs que solo aparecían allí.
    """
    datos = "".join(f"    '{ln}'\n" for trozo in trozos for ln in trozo.split("\n"))
    cabecera = '''#!/usr/bin/env python3
"""
Docta · instalador autocontenido.

Despliega el proyecto completo (módulos, base de conocimiento, corpus y
tokenizer) en ./docta y pasa las comprobaciones. Mismo contenido que las celdas
PAQUETE del notebook de Kaggle. Generado por scripts/construir_kaggle.py: no se
edita a mano.

Uso:
    python docta_instalador.py            # desplegar y comprobar
    python docta_instalador.py --solo     # solo desplegar
    python docta_instalador.py --dest DIR # desplegar en DIR
"""

PAQUETE = [
''' + datos + "]\n\n" + DESPLIEGUE
    return cabecera + INSTALADOR_CUERPO


def construir_zip(crudo: bytes, rutas: list[Path]) -> Path:
    """
    El proyecto como zip, para la vía Dataset de Kaggle.

    Es el plan B y conviene que exista. El notebook autocontenido tiene una
    ventaja clara —no hace falta subir nada— y un coste: pesa 200 KB de base64
    que el editor de Kaggle digiere mal. Para quien prefiera un notebook de 8 KB
    y legible de arriba abajo, se sube este zip como Dataset privado, se adjunta
    al notebook, y listo:

        Add Input → Search → [tu dataset] → y en la celda:
            import sys; sys.path.insert(0, "/kaggle/input/docta")
            from retriever import Retriever, responder

    Mismo código, dos formas de llevarlo a Kaggle. Ninguna necesita credenciales
    dentro del notebook: el Dataset se sube con la cuenta de cada uno, desde el
    navegador o con kaggle.json en su propia máquina.
    """
    import zipfile
    ruta = SALIDA / "docta_dataset_kaggle.zip"
    with zipfile.ZipFile(ruta, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in rutas:
            z.write(p, p.relative_to(RAIZ))
    return ruta


PORTADA_SEPARADO = """# Docta · entrenamiento incremental

Este notebook **no lleva el proyecto dentro**. Lo lee de dos datasets que se
adjuntan con *Add Input*:

| dataset | qué lleva | cuándo cambia |
|---|---|---|
| `docta-codigo` | la arquitectura: módulos, scripts y tests | cuando hay un bug o una mejora |
| `docta-conocimiento` | los datos: fichas curadas, corpus, sft, tokenizer | cada vez que Docta aprende algo |

Separarlos es lo que permite **no volver a empezar de cero**. Añades fichas al
dataset de conocimiento, reanudas el entrenamiento desde el checkpoint anterior, y
listo. La arquitectura no se toca y no se resube.

---

### Primera vez

1. Sube `kaggle/docta_codigo.zip` como Dataset → `docta-codigo`
2. Sube `kaggle/docta_conocimiento.zip` como Dataset → `docta-conocimiento`
3. Adjunta los dos a este notebook con *Add Input*
4. *Run → Run all*

### Cada vez que añadas conocimiento

1. Edita o añade fichas en `data/conocimiento/*.jsonl`
2. `python scripts/construir_corpus.py` — regenera corpus y sft
3. `python scripts/versionar_conocimiento.py` — sube la versión y deja huella
4. `python scripts/construir_kaggle.py` — regenera **solo** el zip de conocimiento
5. Sube esa versión nueva del dataset `docta-conocimiento`
6. En el notebook, `REENTRENAR` encuentra el checkpoint y continúa

Kaggle versiona los datasets por su cuenta, así que queda histórico de qué sabía
el modelo en cada momento. Eso no es un adorno: cuando una IA empieza a decir
algo raro, la primera pregunta siempre es *¿con qué se entrenó?*

### Por qué el tokenizer va con el conocimiento y no con el código

Es un archivo de datos, pero tiene una trampa. Si se reentrena, cambia el
vocabulario y con él la forma de los embeddings: **ningún checkpoint anterior
vuelve a cargar**. El manifiesto lo vigila aparte y avisa cuando pasa.

Mientras no se toque, el BPE es a nivel de byte y cualquier texto nuevo en
español ya es codificable sin reentrenar nada. Es justo lo que hace que el
reentrenamiento incremental sea limpio aquí.
"""


def construir_notebook_separado() -> dict:
    """
    El notebook fino: ~8 KB, legible de arriba abajo, sin base64.

    Existe porque el autocontenido tiene un coste real. Aquel pesa 227 KB de blob
    y el editor de Kaggle lo digiere mal. Este no lleva nada dentro, así que se
    abre, se lee y se edita sin problema, y el conocimiento se actualiza sin
    volver a generar el notebook.
    """
    celdas = [
        celda_md(PORTADA_SEPARADO, id="portada"),
        celda_codigo(CARGAR_DATASETS, id="montar"),
        celda_md("## La puerta de ignorancia\n\nNo necesita entrenar nada: "
                 "funciona con la base curada tal cual viene en el dataset.",
                 id="md-puerta"),
        celda_codigo(DEMO, id="puerta"),
        celda_md("## Comprobaciones", id="md-tests"),
        celda_codigo(TESTS, id="tests"),
        celda_md("## Reentrenar", id="md-reentrenar"),
        celda_codigo(REENTRANAR, id="reentrenar"),
        celda_md("## Preguntar", id="md-preguntar"),
        celda_codigo(CIERRE, id="preguntar"),
        celda_md(ARQUITECTURA, id="arquitectura"),
    ]
    return {
        "cells": celdas,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def escribir_zip(rutas: list[Path], destino: Path) -> Path:
    """Zip determinista: mismo contenido, mismo archivo byte a byte."""
    import zipfile
    destino.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in rutas:
            info = zipfile.ZipInfo(str(p.relative_to(RAIZ)), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, p.read_bytes())
    return destino


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--autocontenido", action="store_true",
                    help="genera TAMBIÉN el notebook con todo dentro (227 KB)")
    ap.add_argument("--solo", choices=["codigo", "conocimiento"], default=None,
                    help="regenera solo un paquete. Es lo habitual al añadir fichas.")
    a = ap.parse_args(argv)

    SALIDA.mkdir(exist_ok=True)
    producidos = []

    # ------------------------------------------------------------- paquetes
    rutas_codigo = rutas_de(PAQUETE_CODIGO)
    rutas_datos = rutas_de(PAQUETE_CONOCIMIENTO)

    if a.solo in (None, "codigo"):
        producidos.append(escribir_zip(rutas_codigo, SALIDA / "docta_codigo.zip"))
    if a.solo in (None, "conocimiento"):
        producidos.append(escribir_zip(rutas_datos, SALIDA / "docta_conocimiento.zip"))

    # ------------------------------------------------- notebook separado
    if a.solo is None:
        ruta_fino = SALIDA / "docta_incremental.ipynb"
        ruta_fino.write_text(
            json.dumps(construir_notebook_separado(), ensure_ascii=False, indent=1),
            encoding="utf-8")
        producidos.append(ruta_fino)

    # --------------------------------------------- notebook autocontenido
    if a.autocontenido:
        crudo, rutas = construir_tar()
        lineas, trozos = partir_base64(crudo)
        nb = construir_notebook(trozos)
        ruta_nb = SALIDA / "docta_kaggle.ipynb"
        ruta_nb.write_text(json.dumps(nb, ensure_ascii=False, indent=1),
                           encoding="utf-8")
        producidos.append(ruta_nb)
        (SALIDA / "docta_instalador.py").write_text(
            construir_instalador(trozos), encoding="utf-8")
        producidos.append(SALIDA / "docta_instalador.py")

        mayor = max(len("".join(c["source"])) for c in nb["cells"])
        print(f"autocontenido : {len(rutas)} archivos · {len(crudo):,} bytes · "
              f"{len(trozos)} celdas · celda mayor {mayor:,} chars")
        if mayor > 40_000:
            # Este umbral no es una constante mágica: es lo que se comprobó que
            # el editor de Kaggle digiere. Con 178.000 chars se bloqueó.
            print(f"  ⚠ celda de {mayor:,} chars: el editor puede bloquearse. "
                  f"Baja TAMANO_TROZO.")

    # ------------------------------------------------------------- resumen
    print()
    for ruta in producidos:
        print(f"-> {ruta.relative_to(RAIZ)}  ({ruta.stat().st_size/1024:.0f} KB)")

    if a.solo is None:
        print(f"\n{len(rutas_codigo)} archivos de código · {len(rutas_datos)} de conocimiento")
        print("\nSube los dos zip como Datasets de Kaggle y adjúntalos al notebook")
        print("docta_incremental.ipynb. Al añadir fichas, regenera solo el de")
        print("conocimiento con: python scripts/construir_kaggle.py --solo conocimiento")
    return 0


if __name__ == "__main__":
    sys.exit(main())