"""
El temario: qué debe saber Docta, y cuánto lleva curado.

De dónde sale. De los dos documentos que se aportaron al proyecto:

  * «Taxonomía Integral de Conocimientos y Competencias Esenciales para el
    Adulto Contemporáneo» — PIAAC y DeSeCo (OCDE), DigComp 2.2, OECD/INFE, las
    diez habilidades de vida de la OMS, UNESCO.
  * «Competencias Académicas Fundamentales y Cultura General Sustancial para la
    Ciudadanía en la Sociedad Contemporánea».

Y de una medición que conviene tener presente. Se pasó un extractor por los dos
documentos buscando frases que afirmen un dato comprobable —una cifra con su
unidad, una fecha, una relación verificable— y el resultado fue **cero de 279
frases**. El mismo extractor, aplicado a las 53 fichas que ya estaban curadas,
encuentra 41 datos en 110 frases, o sea un 37 %. No es que el extractor falle:
es que esos documentos no contienen datos.

Son TEMARIOS, no contenido. Dicen qué hay que saber; no lo saben. «El ciudadano
debe comprender la entropía» no es una ficha, es un apunte de lo que falta por
escribir. Por eso este archivo no alimenta el corpus ni la base de conocimiento:
alimenta la lista de la compra.

Meter el texto de los PDF en `data/corpus.txt` sería activamente dañino, y por
dos vías. En la base de conocimiento, una frase curricular recuperada se le
serviría al usuario como si fuera una respuesta: preguntado por la entropía,
Docta contestaría «el ciudadano debe comprender la irreversibilidad de los
procesos térmicos», que no responde a nada y además sale con una ✓ de
verificación porque está "respaldado" por su propia ficha. Y en el corpus de
pre-entrenamiento enseñaría el registro: Docta acabaría hablando como un informe
de la OCDE, que es justo lo contrario del carácter sobrio y directo que se
decidió para ella.

Uso:
    python scripts/temario.py            # informe de cobertura
    python scripts/temario.py --detalle  # dominio a dominio, qué falta
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from knowledge import BaseConocimiento          # noqa: E402
from tokenizer import normalizar                # noqa: E402

# ---------------------------------------------------------------------------
# El temario.
#
# Cada bloque lleva las palabras que permiten saber si ya hay algo curado sobre
# él. No es una taxonomía decorativa: es lo que convierte este archivo en un
# informe en vez de en una lista de buenas intenciones.
#
# `clave` es el prefijo de tema que se usará en las fichas, para que el índice
# del recuperador y este temario hablen el mismo idioma.
# ---------------------------------------------------------------------------
TEMARIO: list[dict] = [
    dict(
        dominio="Matemáticas y razonamiento cuantitativo",
        fuente="PDF 2 · Pensamiento Estadístico; PDF 1 · Dominio 1 (PIAAC)",
        clave="matematicas",
        bloques=[
            dict(nombre="Estadística descriptiva",
                 marcas=("media", "mediana", "moda", "desviación", "varianza",
                         "percentil", "histograma"),
                 ejemplos=["¿Qué diferencia hay entre media y mediana?",
                           "¿Qué es la desviación típica?"]),
            dict(nombre="Correlación y causalidad",
                 marcas=("correlación", "causalidad", "causa", "sesgo"),
                 ejemplos=["¿Por que dos cosas varíen juntas no implica que una cause la otra?"]),
            dict(nombre="Probabilidad y riesgo",
                 marcas=("probabilidad", "riesgo", "azar", "esperanza"),
                 ejemplos=["¿Qué es el riesgo absoluto frente al relativo?"]),
            dict(nombre="Matemática financiera básica",
                 marcas=("interés compuesto", "interes compuesto", "tasa",
                         "amortización", "inflación", "poder adquisitivo"),
                 ejemplos=["¿Qué es el interés compuesto?", "¿Qué es la inflación?"]),
        ],
    ),
    dict(
        dominio="Ciencias naturales",
        fuente="PDF 2 · Biología, Química, Física",
        clave="ciencias",
        bloques=[
            dict(nombre="Biología y ciencias de la vida",
                 marcas=("célula", "celula", "adn", "gen", "evolución",
                         "evolucion", "fotosíntesis", "ecosistema", "virus",
                         "bacteria", "vacuna"),
                 ejemplos=["¿Qué es el ADN?", "¿Cómo funciona la evolución?",
                           "¿Qué diferencia hay entre virus y bacteria?"]),
            dict(nombre="Química y estructura de la materia",
                 marcas=("átomo", "atomo", "molécula", "molecula", "elemento",
                         "tabla periódica", "ph", "reacción", "reaccion"),
                 ejemplos=["¿Qué es un átomo?", "¿Qué mide el pH?"]),
            dict(nombre="Física y principios universales",
                 marcas=("gravedad", "inercia", "energía", "energia",
                         "termodinámica", "entropía", "newton", "luz",
                         "electromagnetismo", "velocidad de la luz"),
                 ejemplos=["¿Qué es la inercia?",
                           "¿Por qué se conserva la energía?",
                           "¿A qué velocidad va la luz?"]),
            dict(nombre="Tierra y espacio",
                 marcas=("tectónica", "tectonica", "placas", "atmósfera",
                         "planeta", "sistema solar", "clima"),
                 ejemplos=["¿Por qué hay terremotos?",
                           "¿Cuántos planetas hay en el sistema solar?"]),
        ],
    ),
    dict(
        dominio="Historia y procesos globales",
        fuente="PDF 2 · Historia Universal y Procesos Globales",
        clave="historia",
        bloques=[
            dict(nombre="Prehistoria y mundo antiguo",
                 marcas=("neolítico", "neolitico", "agricultura", "imperio",
                         "roma", "grecia", "egipto", "mesopotamia"),
                 ejemplos=["¿Qué cambió con la revolución neolítica?",
                           "¿Cuándo cayó el Imperio romano?"]),
            dict(nombre="Edad Media y edad moderna",
                 marcas=("medieval", "edad media", "feudal", "renacimiento",
                         "descubrimiento", "colón", "colon"),
                 ejemplos=["¿Qué fue el Renacimiento?",
                           "¿En qué año llegó Colón a América?"]),
            dict(nombre="Era de las revoluciones",
                 marcas=("ilustración", "ilustracion", "revolución francesa",
                         "revolucion francesa", "industrial", "liberalismo",
                         "socialismo", "nacionalismo"),
                 ejemplos=["¿Qué fue la Ilustración?",
                           "¿Cuándo fue la Revolución Industrial?"]),
            dict(nombre="Siglo XX",
                 marcas=("guerra mundial", "totalitarismo", "holocausto",
                         "guerra fría", "guerra fria", "nazismo", "fascismo"),
                 ejemplos=["¿Cuándo fue la Segunda Guerra Mundial?",
                           "¿Qué fue la Guerra Fría?"]),
        ],
    ),
    dict(
        dominio="Geografía sistémica y territorial",
        fuente="PDF 2 · Geografía Sistémica; base ya curada",
        clave="geografia",
        bloques=[
            dict(nombre="Geografía física",
                 marcas=("continente", "océano", "oceano", "río", "rio",
                         "montaña", "montana", "desierto", "clima", "placa"),
                 ejemplos=["¿Cuál es el río más largo del mundo?"]),
            dict(nombre="Geografía humana y política",
                 marcas=("capital", "país", "pais", "población", "poblacion",
                         "frontera", "idioma", "ciudad"),
                 ejemplos=["¿Cuál es la capital de Francia?"]),
            dict(nombre="Geopolítica y recursos",
                 marcas=("geopolítica", "geopolitica", "recursos", "agua dulce",
                         "hidrocarburos", "mineral", "soberanía", "soberania"),
                 ejemplos=["¿Por qué el agua dulce es un recurso estratégico?"]),
        ],
    ),
    dict(
        dominio="Lengua, literatura y comunicación",
        fuente="PDF 2 · Lengua y Literatura; Comprensión Lectora; PDF 1 · Dominio 1",
        clave="lengua",
        bloques=[
            dict(nombre="Literatura universal",
                 marcas=("cervantes", "quijote", "shakespeare", "homero",
                         "homero", "garcía márquez", "obra", "novela", "poeta",
                         "escritor", "siglo de oro"),
                 ejemplos=["¿Quién escribió el Quijote?",
                           "¿Quién escribió Cien años de soledad?"]),
            # "español" a secas también hubo que quitarlo: casaba con las seis
            # fichas de países hispanohablantes, que son geografía demográfica, no
            # lengua. Se exigen marcas que solo tienen sentido lingüístico.
            dict(nombre="Lengua española",
                 marcas=("gramática", "gramatica", "sintaxis", "ortografía",
                         "ortografia", "sustantivo", "verbo", "rae",
                         "hablantes de español", "lengua romance", "morfología",
                         "morfologia"),
                 ejemplos=["¿Cuántos hablantes tiene el español?",
                           "¿Qué es un sustantivo?"]),
            # "fuente" y "verificación" estaban aquí y hubo que quitarlas: en la
            # base esas palabras significan la CAPA DE CITAS, no alfabetización
            # mediática, y hacían que dos fichas sobre la ciudad más poblada del
            # mundo y sobre la fiabilidad de Docta contaran como lectura crítica.
            # Es el límite de casar palabras: no distingue "una ficha SOBRE X" de
            # "una ficha QUE MENCIONA X". Cuando una marca tiene dos sentidos en
            # el propio proyecto, no sirve de marca.
            dict(nombre="Lectura crítica y desinformación",
                 marcas=("falacia", "desinformación", "desinformacion",
                         "noticia falsa", "propaganda", "sesgo cognitivo",
                         "leer críticamente", "fake news"),
                 ejemplos=["¿Qué es una falacia?",
                           "¿Cómo se detecta una noticia falsa?"]),
        ],
    ),
    dict(
        dominio="Tecnología y competencia digital",
        fuente="PDF 1 · Dominio 2 (DigComp 2.2); PDF 2 · Competencia Digital e IA",
        clave="tecnologia",
        bloques=[
            dict(nombre="Alfabetización de datos",
                 marcas=("búsqueda", "busqueda", "repositorio", "archivo",
                         "metadato", "navegador"),
                 ejemplos=["¿Qué es un metadato?"]),
            dict(nombre="Ciberseguridad",
                 marcas=("phishing", "contraseña", "contrasena", "cifrado",
                         "malware", "estafa", "ingeniería social", "dos factores"),
                 ejemplos=["¿Qué es el phishing?",
                           "¿Qué es el cifrado de extremo a extremo?"]),
            dict(nombre="Inteligencia artificial",
                 marcas=("inteligencia artificial", "modelo de lenguaje",
                         "red neuronal", "aprendizaje automático", "alucinación",
                         "alucinacion", "entrenamiento", "sesgo algorítmico"),
                 ejemplos=["¿Qué es una red neuronal?",
                           "¿Qué es una alucinación de un modelo de lenguaje?"]),
            dict(nombre="Huella e identidad digital",
                 marcas=("huella digital", "reputación", "reputacion",
                         "datos personales", "cookies", "privacidad"),
                 ejemplos=["¿Qué es la huella digital?"]),
        ],
    ),
    dict(
        dominio="Economía y finanzas personales",
        fuente="PDF 1 · Dominio 3 (OECD/INFE); PDF 2 · Matemáticas Financieras",
        clave="economia",
        bloques=[
            dict(nombre="Conceptos económicos básicos",
                 marcas=("inflación", "inflacion", "pib", "oferta", "demanda",
                         "banco central", "tipo de interés", "desempleo"),
                 ejemplos=["¿Qué es la inflación?", "¿Qué es el PIB?"]),
            dict(nombre="Productos y servicios financieros",
                 marcas=("hipoteca", "préstamo", "prestamo", "seguro",
                         "fondo de inversión", "acciones", "bono", "tarjeta",
                         "interés fijo", "interés variable"),
                 ejemplos=["¿Qué es una hipoteca?",
                           "¿Qué diferencia hay entre una acción y un bono?"]),
            dict(nombre="Presupuesto y ahorro",
                 marcas=("presupuesto", "ahorro", "fondo de emergencia",
                         "endeudamiento", "jubilación", "jubilacion", "pensión",
                         "pension"),
                 ejemplos=["¿Qué es un fondo de emergencia?"]),
            dict(nombre="Fiscalidad básica",
                 marcas=("iva", "irpf", "impuesto", "declaración", "declaracion",
                         "retención", "retencion", "nómina", "nomina"),
                 ejemplos=["¿Qué es el IVA?", "¿Qué es el IRPF?"]),
        ],
    ),
    dict(
        dominio="Ética, filosofía y ciudadanía",
        fuente="PDF 1 · Dominio 5 (DeSeCo, UNESCO); temario original del proyecto",
        clave="etica",
        bloques=[
            dict(nombre="Filosofía: corrientes y autores",
                 marcas=("sócrates", "socrates", "platón", "platon",
                         "aristóteles", "aristoteles", "kant", "nietzsche",
                         "estoicismo", "existencialismo", "ética", "etica",
                         "moral", "utilitarismo"),
                 ejemplos=["¿Qué es el estoicismo?",
                           "¿Qué diferencia hay entre ética y moral?"]),
            dict(nombre="Derechos humanos y marco constitucional",
                 marcas=("derechos humanos", "constitución", "constitucion",
                         "declaración universal", "democracia", "sufragio",
                         "separación de poderes", "estado de derecho"),
                 ejemplos=["¿Cuándo se firmó la Declaración Universal?",
                           "¿Qué es la separación de poderes?"]),
            dict(nombre="Sostenibilidad",
                 marcas=("sostenibilidad", "cambio climático",
                         "cambio climatico", "emisiones", "renovable",
                         "biodiversidad", "huella de carbono"),
                 ejemplos=["¿Qué es la huella de carbono?",
                           "¿Qué causa el cambio climático?"]),
            dict(nombre="Interculturalidad y convivencia",
                 marcas=("intercultural", "diversidad", "inclusión",
                         "inclusion", "equidad", "empatía", "empatia"),
                 ejemplos=["¿Qué diferencia hay entre igualdad y equidad?"]),
        ],
    ),
    dict(
        dominio="Metacognición y límites propios",
        fuente="decisión de diseño del proyecto; PDF 1 · Dominio 4 (OMS)",
        clave="meta",
        bloques=[
            dict(nombre="Lo que Docta no sabe y por qué",
                 marcas=("no lo sé", "no se", "límite", "limite", "incertidumbre",
                         "fuente", "verificado", "certeza"),
                 ejemplos=["¿Por qué dices que no lo sabes?"]),
        ],
    ),
]


# ---------------------------------------------------------------------------
# Cobertura
# ---------------------------------------------------------------------------
def _marcador(marca: str):
    """
    Expresión regular de palabra completa, sobre texto ya normalizado.

    Imprescindible y no un refinamiento. La primera versión hacía
    `if marca in texto`, que es búsqueda por subcadena, y el informe salía
    inflado de forma espectacular: "gen" casaba dentro de "gente" y de "general"
    (13 fichas de geografía contaban como biología), "iva" dentro de
    "legislativa" (Canberra y Pretoria salían como fiscalidad), "bono" dentro de
    "abono". El temario decía que había un 48 % cubierto cuando casi todo era
    ruido.

    Es literalmente el mismo fallo que ya se había corregido una vez en el
    recuperador, donde "iva" vetaba "recursiva". Se volvió a cometer aquí por
    escribir la comprobación rápida en vez de reutilizar la buena. La lección no
    es "ten cuidado con las subcadenas": es que cuando un proyecto ya resolvió un
    problema, se busca la solución existente antes de escribir otra.
    """
    return re.compile(r"(?<![0-9a-z])" + re.escape(normalizar(marca)) + r"(?![0-9a-z])")


def cobertura(base: BaseConocimiento) -> list[dict]:
    """
    Cruza el temario con lo que hay curado.

    Un bloque se considera cubierto si alguna ficha de la base menciona alguna de
    sus marcas, por palabra completa. Es una heurística y se dice: que una ficha
    contenga la palabra "media" no garantiza que explique la media aritmética.
    Pero para saber por dónde empezar a curar basta, y es mucho mejor que ir a
    ojo — siempre que no mienta, que es lo que hacía antes de arreglar lo de las
    subcadenas.
    """
    hechos = [(h, normalizar(h.texto_completo())) for h in base]
    salida = []
    for dom in TEMARIO:
        bloques = []
        for b in dom["bloques"]:
            marcas = [_marcador(m) for m in b["marcas"]]
            que = [h for h, txt in hechos
                   if any(rx.search(txt) for rx in marcas)]
            bloques.append({**b, "n_hechos": len(que),
                            "ids": [h.id for h in que][:6]})
        total = sum(x["n_hechos"] for x in bloques)
        salida.append({**dom, "bloques": bloques,
                       "cubiertos": sum(1 for x in bloques if x["n_hechos"]),
                       "total_bloques": len(bloques),
                       "hechos": total})
    return salida


def informar(detalle: bool = False) -> int:
    base = BaseConocimiento.cargar()
    doms = cobertura(base)

    bloques_tot = sum(d["total_bloques"] for d in doms)
    cubiertos = sum(d["cubiertos"] for d in doms)

    print("=" * 76)
    print(f"TEMARIO · {len(doms)} dominios · {bloques_tot} bloques")
    print(f"base actual: {len(base)} hechos · cubiertos {cubiertos}/{bloques_tot} "
          f"bloques ({cubiertos/bloques_tot:.0%})")
    print("cobertura APROXIMADA: casar palabras no distingue una ficha que TRATA"
          " de un tema")
    print("de una que lo MENCIONA de pasada. Sirve para saber por dónde empezar,"
          " no como verdad.")
    print("=" * 76)
    print()

    ancho = max(len(d["dominio"]) for d in doms)
    for d in doms:
        barra = "█" * d["cubiertos"] + "·" * (d["total_bloques"] - d["cubiertos"])
        estado = "completo" if d["cubiertos"] == d["total_bloques"] else (
            "parcial" if d["cubiertos"] else "VACÍO")
        print(f"  {d['dominio']:<{ancho}}  {barra}  "
              f"{d['cubiertos']}/{d['total_bloques']}  {estado:>9}  "
              f"{d['hechos']} hechos")
        if detalle:
            print(f"      fuente: {d['fuente']}")
            for b in d["bloques"]:
                marca = "✓" if b["n_hechos"] else " "
                print(f"      {marca} {b['nombre']} — {b['n_hechos']} hechos")
                if not b["n_hechos"]:
                    for e in b["ejemplos"][:2]:
                        print(f"          falta por ejemplo: {e}")
            print()

    print()
    vacios = [d["dominio"] for d in doms if d["cubiertos"] == 0]
    if vacios:
        print(f"dominios sin una sola ficha: {len(vacios)}")
        for v in vacios:
            print(f"  · {v}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--detalle", action="store_true",
                    help="dominio a dominio, con lo que falta por curar")
    a = ap.parse_args(argv)
    return informar(a.detalle)


if __name__ == "__main__":
    sys.exit(main())
