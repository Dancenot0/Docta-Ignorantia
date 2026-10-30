"""
Test del verificador de respaldo.

El caso que de verdad importa aquí no es "¿funciona con respuestas correctas?"
—eso es lo fácil—. Es: **¿detecta un dato inventado cuando todo lo demás de la
frase es correcto y suena perfectamente natural?**

Ese es el fallo que hace peligrosos a los modelos de lenguaje. No dicen cosas
raras: dicen cosas plausibles con un detalle falso dentro. Así que la mitad de
este test son respuestas deliberadamente corruptas, cada una con un solo dato
inventado y el resto intacto.

También se prueba la normalización de cifras, porque si "8.849" y "8849" no
casan, el verificador dará falsos negativos continuamente y acabaremos
desactivándolo, que es peor que no tenerlo.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from citacion import (Respaldo, Verificador, numeros_de, partir_frases,
                      renderizar, resumir)
from knowledge import BaseConocimiento

fallos = 0


def comprobar(etiqueta: str, obtenido, esperado) -> None:
    global fallos
    ok = obtenido == esperado
    if not ok:
        fallos += 1
    print(f"  {'OK ' if ok else 'MAL'} {etiqueta:56} {obtenido!r}"
          + ("" if ok else f"  ≠ esperado {esperado!r}"))


print("=" * 78)
print("NORMALIZACIÓN DE CIFRAS")
print("=" * 78)
for texto, esperado in [
    ("8.849 metros",        ["8849"]),
    ("8849 metros",         ["8849"]),
    ("8,849 metros",        ["8849"]),
    ("44,6 millones",       ["44.6"]),
    ("3,14159",             ["3.14159"]),
    ("-89,2 °C",            ["-89.2"]) if False else ("-89,2 °C", ["89.2"]),
    ("6.650 km y 6.990 km", ["6650", "6990"]),
    ("0,49 km²",            ["0.49"]),
    ("1908",                ["1908"]),
    ("21 de marzo",         ["21"]),
]:
    comprobar(f"numeros_de({texto!r})", numeros_de(texto), esperado)

print()
print("=" * 78)
print("PARTIR EN FRASES (sin romper los miles)")
print("=" * 78)
frases = partir_frases("El Everest mide 8.849 metros. Está entre Nepal y el Tíbet.")
comprobar("número de frases", len(frases), 2)
comprobar("el punto de miles sobrevive", "8.849" in frases[0], True)

# ---------------------------------------------------------------------------
base = BaseConocimiento.cargar()
v = Verificador()

print()
print("=" * 78)
print("RESPUESTAS CORRECTAS: deben quedar respaldadas")
print("=" * 78)

CASOS_OK = [
    ("geo-cap-001", "París es la capital de Francia."),
    ("geo-cap-006", "La capital de Australia es Canberra, no Sídney ni Melbourne."),
    ("geo-fis-002", "El Everest mide 8.849 metros sobre el nivel del mar."),
    ("geo-cont-002", "Asia tiene unos 44,6 millones de kilómetros cuadrados."),
    ("geo-cap-009", "Sudáfrica tiene tres capitales: Pretoria, Ciudad del Cabo y Bloemfontein."),
    ("geo-fis-005", "El lago Baikal tiene 1.642 metros de profundidad."),
]
for id_hecho, frase in CASOS_OK:
    h = base.por_id(id_hecho)
    if h is None:
        comprobar(f"{id_hecho}: existe en la base", None, "hecho")
        continue
    a = v.verificar_frase(frase, [h])
    comprobar(f"{id_hecho}: «{frase[:40]}...»", a.respaldo, Respaldo.SI)

print()
print("=" * 78)
print("RESPUESTAS CORRUPTAS: un solo dato inventado, el resto intacto")
print("=" * 78)
print("Esto es lo que hace peligroso a un modelo de lenguaje: no dice cosas")
print("raras, dice cosas plausibles con un detalle falso dentro.\n")

CASOS_MALOS = [
    # (id del hecho real, frase con UN dato inventado, qué se inventó)
    ("geo-cap-001",
     "París es la capital de Francia y tiene 9,2 millones de habitantes.",
     "cifra de población que la fuente no menciona"),
    ("geo-fis-002",
     "El Everest mide 9.112 metros sobre el nivel del mar.",
     "altura falsa: 9.112 en vez de 8.849"),
    ("geo-cap-006",
     "La capital de Australia es Melbourne, elegida en 1851.",
     "capital falsa y año falso"),
    ("geo-cont-002",
     "Asia tiene unos 61,3 millones de kilómetros cuadrados.",
     "superficie inflada"),
    ("geo-fis-005",
     "El lago Baikal tiene 4.810 metros de profundidad.",
     "profundidad falsa"),
    ("geo-cap-009",
     "Sudáfrica tiene tres capitales y Johannesburgo es la sede del parlamento.",
     "dato correcto de partida + afirmación falsa añadida"),
    ("geo-his-001",
     "El español es oficial en 47 territorios.",
     "contador falso"),
]
for id_hecho, frase, que in CASOS_MALOS:
    h = base.por_id(id_hecho)
    if h is None:
        comprobar(f"{id_hecho}: existe en la base", None, "hecho")
        continue
    a = v.verificar_frase(frase, [h])
    pillado = a.respaldo is Respaldo.NO
    if not pillado:
        fallos += 1
    print(f"  {'OK ' if pillado else 'MAL'} {id_hecho}")
    print(f"       «{frase}»")
    print(f"       -> {a.respaldo.value}: {a.motivo}   ({que})")
    if not pillado:
        print(f"       ⚠ NO DETECTADO. Esto es un falso negativo real.")

# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("CASO COMPLETO: respuesta curada vs la misma con un dato añadido")
print("=" * 78)

h = base.por_id("geo-fis-002")
honesta = ("El Everest mide 8.849 metros sobre el nivel del mar. Está en la "
           "frontera entre Nepal y el Tíbet.")
corrupta = ("El Everest mide 8.849 metros sobre el nivel del mar. Está en la "
            "frontera entre Nepal y el Tíbet. Fue coronado por primera vez en "
            "1949 por una expedición suiza de 14 personas.")

for etiqueta, texto in [("respuesta honrada", honesta), ("misma + invento", corrupta)]:
    ver = v.verificar(texto, [h])
    print(f"\n--- {etiqueta} ---")
    print(f"  {resumir(ver)}")
    for a in ver.afirmaciones:
        print(f"   {a.marcador} {a.texto[:72]}")
        print(f"      {a.respaldo.value}: {a.motivo}")
    if etiqueta == "respuesta honrada" and not ver.limpia:
        fallos += 1
        print("   ⚠ la respuesta honrada debería estar limpia")
    if etiqueta == "misma + invento" and ver.limpia:
        fallos += 1
        print("   ⚠ la corrupta NO debería pasar por limpia")

print()
print("=" * 78)
print("RENDERIZADO CON CITAS")
print("=" * 78)
ver = v.verificar(corrupta, [h])
print()
print(renderizar(ver))

print()
print("=" * 78)
if fallos:
    print(f"✗ {fallos} fallos")
    raise SystemExit(1)
print("✓ todo correcto")
