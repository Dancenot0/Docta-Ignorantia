"""
Retrieval + PUERTA DE IGNORANCIA.

Este módulo es el que hace que el proyecto cumpla su promesa principal: que la
IA no lo sepa todo, y que lo diga cuando no lo sabe.

Dos piezas:

1. Un recuperador léxico (TF-IDF + bonus por coincidencia de frases). Sin
   vectores ni embeddings ni modelos externos: funciona, es rápido, es
   explicable y no depende de nada que no hayamos escrito nosotros. Se puede
   sustituir después por búsqueda densa sin tocar el resto del sistema.

2. La puerta de ignorancia. Tres condiciones independientes, y basta con que
   falle UNA para que la IA se niegue a responder con un dato:
      a) ¿hay evidencia?      -> el mejor resultado supera el umbral de similitud
      b) ¿es de fiar?         -> la certeza curada del hecho supera el umbral
      c) ¿está dentro de lo que sé? -> el tema no está en la lista negra

   La decisión final NO es "contesto o me callo", es "contesto con este grado
   de apoyo". Y el grado se le pasa al generador para que ajuste el tono: no es
   lo mismo afirmar que dudar.

Por qué esto importa más que el modelo neural: un generador de lenguaje puede
escribir una mentira con una gramática impecable. Sin una puerta que le corte
el paso cuando no hay evidencia detrás, tienes un loro muy elocuente. Con la
puerta, tienes algo que se comporta como una persona honesta.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional, Sequence

if TYPE_CHECKING:      # solo para anotaciones; importarlo de verdad
    from citacion import Verificacion   # cerraría un ciclo con citacion
from knowledge import BaseConocimiento, Hecho
from tokenizer import normalizar

# Palabras que no aportan nada a la búsqueda.
STOPWORDS = frozenset("""
a al algo algun alguna algunas algunos ante antes aquel aquella aquellas aquello
aquellos aqui asi aun aunque con como contra cual cuales cualquier cuando cuanto
de del ella ellas ello ellos en entre era eran eres es esa esas ese eso esos esta
estaba estan estar estas este esto estos fue fueron ha hace hacia han hasta hay la
las le les lo los mas me mi mis mucho muchos muy nada ni no nos nosotros o os otra
otras otro otros para pero poco por porque que quien quienes se sea ser si sin sino
sobre solo son su sus tambien tan te tiene tienen to toda todas todo todos tu tus un
una unas uno unos usted ustedes va vamos y ya yo cual cualesquiera
cual es cual es la cual de que de quien como donde porque cuando
""".split())

# Temas sobre los que la IA NO debe opinar aunque encuentre algo parecido.
# La gracia es que esto es explícito y auditable, no un comportamiento emergente.
LISTA_NEGRA = frozenset({
    "medicina", "diagnostico", "farmacos", "legal", "derecho", "fiscal",
    "finanzas", "inversion", "programacion", "codigo", "actualidad", "noticias",
})


@dataclass
class Candidato:
    hecho: Hecho
    similitud: float
    apoyo: float                 # similitud × certeza  -> confianza final
    motivos: list[str] = field(default_factory=list)
    ents_cubiertas: int = 0      # entidades de la pregunta que aparecen en el hecho
    ents_total: int = 0
    ents_faltan: list[str] = field(default_factory=list)

    @property
    def cubre_entidades(self) -> bool:
        """True si no había entidades que cubrir, o si las cubre todas."""
        return self.ents_total == 0 or self.ents_cubiertas == self.ents_total


@dataclass
class RespuestaRetrieval:
    """Resultado de consultar la base, con la decisión de la puerta incluida."""
    consulta: str
    sabe: bool                   # ¿debemos afirmar algo?
    apoyo: float                 # 0..1, confianza del mejor candidato
    mejor: Optional[Candidato]
    candidatos: list[Candidato]
    motivo_rechazo: Optional[str] = None
    # Verificación de la respuesta contra la evidencia. None si no se llegó a
    # responder. Es lo que permite CITAR: sin esto el sistema tendría una
    # respuesta y una lista de fichas, pero no sabría qué parte de la respuesta
    # sostiene cada ficha.
    verificado: Optional["Verificacion"] = None

    @property
    def grado(self) -> str:
        """Cómo debe hablar la IA según el apoyo que tenga."""
        if not self.sabe:
            return "no_lo_se"
        if self.apoyo >= 0.75:
            return "seguro"
        if self.apoyo >= 0.5:
            return "probable"
        return "dudoso"

    def contexto(self, max_hechos: int = 2) -> str:
        """Texto de apoyo que se le inyecta al generador."""
        if not self.candidatos:
            return ""
        trozos = []
        for c in self.candidatos[:max_hechos]:
            trozos.append(f"- {c.hecho.respuesta}")
            if c.hecho.nota:
                trozos.append(f"  (aviso: {c.hecho.nota})")
        return "\n".join(trozos)


# ---------------------------------------------------------------------------
# Índice léxico
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[a-z0-9ñü]+")

# Palabras en mayúscula que NO son entidades aunque lo parezcan: aparecen en
# mayúscula solo por empezar la frase o por ser nombre de lengua o gentilicio.
# Sin esta lista, "¿Qué país...?" metería "Qué" como entidad obligatoria y no
# casaría con nada nunca.
NO_ENTIDADES = frozenset("""
qué que cuál cual quién quien cómo como dónde donde cuándo cuando cuánto cuanto
cuánta cuanta cuántos cuantos por qué porque pero si no y o u a al de del la
el los las un una unos unas en con sin sobre bajo entre hasta desde para
españa europa américa africa asia oceanía oceania antártida antartida
español española españoles castellano inglés inglesa francés francesa portugués
portuguesa alemán alemana italiano italiana chino japonés japonesa ruso rusa
árabe hindú hoy ayer mañana
""".split())

_ENTIDAD_RE = re.compile(r"\b([A-ZÁÉÍÓÚÑÜ][a-záéíóúñü0-9]+)\b")


def entidades(texto: str) -> set[str]:
    """
    Nombres propios de una pregunta, normalizados.

    Para qué sirve: la similitud léxica por bolsa de palabras no distingue el
    SUJETO de la pregunta. "¿Qué idioma se habla en Bután?" se parece mucho a un
    hecho sobre los idiomas de América del Sur, porque comparten "idioma" y
    "habla". Y el resultado es que la IA contesta sobre Brasil cuando le han
    preguntado por Bután — con total fluidez y un dato perfectamente correcto
    sobre un país que nadie ha mencionado. Es el fallo más peligroso que puede
    tener este sistema, porque no se parece a un error: se parece a una
    respuesta.

    Exigir que las entidades de la pregunta aparezcan en el hecho lo corta de
    raíz. No es una solución perfecta (una entidad puede aparecer de pasada),
    pero convierte un fallo silencioso en uno visible.
    """
    crudo = re.sub(r"^[¿?¡!\s]+", "", texto.strip())
    crudo = re.sub(r"^\S+", lambda m: m.group(0).lower(), crudo, count=1)
    salida = set()
    for w in _ENTIDAD_RE.findall(crudo):
        n = normalizar(w)
        if n and n not in NO_ENTIDADES and len(n) > 1:
            salida.add(n)
    return salida


def cobertura_entidades(consulta: str, hecho: Hecho) -> tuple[int, int, list[str]]:
    """
    Cuántas entidades de la pregunta aparecen en el hecho.
    Devuelve (encontradas, totales, las que faltan).
    """
    ents = entidades(consulta)
    if not ents:
        return 0, 0, []
    donde = normalizar(hecho.texto_completo())
    faltan = [e for e in ents if e not in donde]
    return len(ents) - len(faltan), len(ents), faltan


def tokens(texto: str) -> list[str]:
    """Palabras normalizadas y sin stopwords, para indexar."""
    crudo = normalizar(texto)
    return [t for t in _TOKEN_RE.findall(crudo) if t not in STOPWORDS and len(t) > 1]


def bigramas(texto: str) -> list[str]:
    """
    Pares de palabras consecutivas: detectar una frase entera pesa más.

    Estos van CON stopwords y los de `citacion.pares_de` sin ellos, y no es un
    descuido. Aquí sirven para puntuar el parecido entre una consulta y un
    hecho, donde "capital de francia" pesa más que "capital francia"; allí sirven
    para verificar una afirmación, donde los pares vacíos solo hacen ruido. Dos
    funciones parecidas con nombres distintos es mejor que una función que hace
    dos trabajos y hace mal uno.
    """
    ts = tokens(texto)
    return [f"{a}_{b}" for a, b in zip(ts, ts[1:])]


# ---------------------------------------------------------------------------
# Detección de temas vetados EN LA PROPIA PREGUNTA.
#
# Esto no es un adorno. La lista negra anterior solo miraba el tema del hecho
# recuperado, y eso deja un agujero: si una pregunta de medicina se parece en
# palabras a un hecho de geografía, el recuperador devuelve geografía, el tema
# no está vetado, y la puerta responde "no llega para contestarte" en vez de
# "de esto no opino, pregúntale a un médico". La respuesta sigue siendo no
# dar el dato, así que no es catastrófico, pero es la respuesta equivocada:
# no le dice al usuario que el tema es peligroso ni le señala a quién ir.
#
# Regla de diseño: el veto se decide sobre la PREGUNTA, antes de mirar qué hay
# en la base. Lo que la IA sabe no debe influir en lo que se niega a opinar.
# ---------------------------------------------------------------------------
SEÑALES_VETO: dict[str, tuple[str, ...]] = {
    # Cada señal es una PALABRA COMPLETA o una FRASE de varias palabras.
    # Nunca una subcadena suelta: "iva" es subcadena de "recursiva", "la" lo es
    # de medio diccionario, y así se veta en falso la mitad de las preguntas.
    "medicina": (
        "dosis", "mg", "miligramos", "paracetamol", "ibuprofeno", "aspirina",
        "antibiótico", "antibiotico", "medicamento", "medicamentos", "pastilla",
        "pastillas", "fármaco", "farmaco", "receta", "síntoma", "sintoma",
        "síntomas", "sintomas", "diagnóstico", "diagnostico", "tratamiento",
        "médico", "medico", "médica", "medicina", "enfermedad", "enfermedades",
        "inyección", "inyeccion", "vacuna", "vacunas", "embarazo", "lactante",
        "bebé", "bebe", "quimioterapia", "análisis de sangre", "me duele",
        "qué puedo tomar", "cuánto puedo tomar", "tiene cura",
    ),
    "legal": (
        "demandar", "denunciar", "juicio", "abogado", "abogada", "contrato",
        "cláusula", "clausula", "ilegal", "legal", "ley", "leyes",
        "legislación", "legislacion", "tribunal", "juez", "multa", "sanción",
        "sancion", "herencia", "testamento", "divorcio", "custodia", "demanda",
    ),
    "fiscal": (
        "impuesto", "impuestos", "hacienda", "irpf", "iva", "autónomo",
        "autonomo", "cotizar", "declaración de la renta", "deducción fiscal",
        "factura", "facturas",
        # Y aquí lo mismo, pero con un matiz: "deducción" y "declaración" a secas
        # son más anchas que las frases anteriores, y podrían rozar preguntas
        # legítimas («¿qué es una declaración de independencia?»). Se añaden
        # porque ahora es SEGURO añadirlas: el filtro de intención ya distingue
        # «¿qué es una deducción fiscal?» —concepto, se contesta— de «¿qué
        # deducción me aplica?» —consejo, se veta. Antes, ampliar la lista de
        # palabras significaba vetar más en falso. Ahora significa acertar más.
        "deducción", "deduccion", "deducciones", "declaración", "declaracion",
        "declaraciones", "tributario", "tributaria", "retención", "retencion",
    ),
    "finanzas": (
        "invertir", "inversión", "inversion", "inversiones", "criptomoneda",
        "criptomonedas", "bitcoin", "acciones", "hipoteca", "préstamo",
        "prestamo", "rentabilidad", "trading", "apostar", "fondos de inversión",
        "merece la pena comprar", "comprar acciones",
        # Lo mismo que en los otros temas: la gente no pregunta «¿invertir es
        # bueno?», pregunta «¿invierto ahora?». Una lista que solo lleva
        # infinitivos se pierde justo la formulación más parecida a la real.
        "invierto", "invierta", "invierte", "invertiré", "invertire",
        "compro", "vendo", "ahorro", "ahorrar",
    ),
    # Aquí estaban los nombres de lenguajes (python, java, rust...). Se quitaron
    # porque son ambiguos: "¿cuál es la isla más grande de Java?" es una
    # pregunta de geografía perfectamente legítima y quedaba vetada. Los
    # nombres propios de lenguajes también son nombres de lugares, islas y
    # personas. Se detecta por los verbos y sustantivos inequívocos.
    "programacion": (
        "código", "codigo", "programar", "programación", "programacion",
        "compila", "compilar", "depurar", "debug", "librería", "libreria",
        "algoritmo", "algoritmos", "recursiva", "recursivo", "función en",
        "implementar", "api", "script", "variable global", "bucle for",
        # Conjugaciones. "programar" estaba y "programo" no, y una lista que solo
        # lleva infinitivos se pierde la mitad de las preguntas reales, que es
        # como la gente habla: «¿cómo hackeo...?», «¿cómo programo...?».
        "hackear", "hackeo", "hackea", "crackear", "programo", "programas",
    ),
    "actualidad": (
        "hoy", "ayer", "últimas noticias", "ultimas noticias", "quién ganó",
        "quien gano", "elecciones", "mundial", "presidente actual",
        "precio actual", "cotización de hoy", "esta semana", "quién es el actual",
    ),
}



# ---------------------------------------------------------------------------
# La INTENCIÓN de la pregunta.
#
# Esto es lo que separa una cultura general útil de un asesoramiento peligroso,
# y el veto por palabras no podía verlo. Bloqueaba igual «¿qué es el IVA?» —que
# es algo que cualquier adulto sabe y que el temario de competencias del adulto
# pide expresamente— que «¿cómo declaro el IVA siendo autónomo?» —que es un
# trámite personal donde un error cuesta dinero. Las dos contienen "iva".
#
# El criterio no está en el TEMA sino en a quién se refiere la pregunta:
#
#   CONCEPTO  pregunta por el mundo. «¿qué es el interés compuesto?»,
#             «¿cómo funciona la inflación?». Es conocimiento general y se puede
#             contestar si la base lo tiene.
#   CONSEJO   pregunta por la situación de quien habla. «¿me conviene
#             refinanciar?», «¿qué dosis le doy a mi hijo?». Lleva primera o
#             segunda persona, un posesivo, o un verbo de decisión aplicado al
#             caso propio. De eso no se opina.
#   ABSOLUTO  nunca, se pregunte como se pregunte. Cantidades de un fármaco para
#             una persona concreta, o instrucciones para causar daño. Ni como
#             concepto: la respuesta a «¿cuál es la dosis letal?» no es cultura
#             general, es un instructivo.
#
# Regla que se mantiene del diseño anterior: esto se decide sobre la PREGUNTA,
# antes de mirar qué hay en la base. Lo que la IA sabe no debe influir en lo que
# se niega a opinar.
# ---------------------------------------------------------------------------
class Intencion(Enum):
    CONCEPTO = "concepto"
    CONSEJO = "consejo"
    ABSOLUTO = "absoluto"


# Marcas de que la pregunta va sobre la situación de quien la hace.
_PRIMERA_PERSONA = frozenset({
    "me", "mi", "mis", "yo", "nos", "nosotros", "nosotras", "conmigo",
    "te", "tu", "tus", "ti", "contigo", "le", "les", "su", "sus",
})

# Verbos y giros de decisión personal: piden que se les diga QUÉ HACER.
_VERBOS_DECISION = frozenset({
    "conviene", "debo", "deberia", "puedo", "podria", "tengo", "necesito",
    "quiero", "hago", "elijo", "compro", "vendo", "invierto", "contrato",
    "declaro", "demandar", "denuncio", "tomo", "doy", "le doy", "empiezo",
    "recomiendas", "aconsejas", "suggest",
})

# Giros que, juntos, son inequívocamente una petición de consejo.
_GIROS_CONSEJO = (
    "que hago", "qué hago", "que deberia", "qué debería", "me recomiendas",
    "me conviene", "merece la pena", "vale la pena", "como puedo", "cómo puedo",
    "que puedo tomar", "qué puedo tomar", "cuanto puedo tomar",
    "cuánto puedo tomar", "me duele", "tengo un", "tengo una", "mi hijo",
    "mi hija", "mi bebe", "mi bebé", "mi madre", "mi padre", "mi vecino",
    "mi caso", "mi situacion", "mi situación", "para mi", "para mí",
    "en mi caso", "que me pasa", "qué me pasa", "deberia contratar",
    "debería contratar", "me compensa", "como declaro", "cómo declaro",
)

# Se pregunta por el mundo: definiciones, funcionamiento, historia, tipos.
_GIROS_CONCEPTO = (
    "que es", "qué es", "que son", "qué son", "como funciona", "cómo funciona",
    "que significa", "qué significa", "para que sirve", "para qué sirve",
    "en que consiste", "en qué consiste", "que tipos", "qué tipos",
    "cual es la diferencia", "cuál es la diferencia", "quien invento",
    "quién inventó", "quien descubrio", "quién descubrió", "cuando se creo",
    "cuándo se creó", "cuando surgio", "cuándo surgió", "por que existe",
    "por qué existe", "que establece", "qué establece", "que define",
    "qué define", "quien lo regula", "quién lo regula", "como se llama",
    "cómo se llama", "cual es el origen", "cuál es el origen",
    "que dice la ley", "qué dice la ley", "que es lo que",
)

# Nunca, se formule como se formule. Se comprueba ANTES que nada lo demás: si una
# pregunta cae aquí, da igual que también parezca un concepto.
#
# Cada entrada es una TUPLA de fragmentos que deben aparecer TODOS. Ojo con la
# coma final, y esto no es un detalle estético: `("letal de")` NO es una tupla, es
# un string entre paréntesis, y entonces el bucle itera letra a letra y
# `all(letra in texto)` es verdad en casi cualquier frase. El veto absoluto se
# dispararía con todo. La primera versión tenía ese fallo en siete entradas y lo
# cazó un test de "¿qué es el paracetamol?" saliendo como absoluto. De ahí la
# comprobación de más abajo: que falle al importar es mil veces mejor que un veto
# que se vuelve loco en silencio.
_ABSOLUTO = (
    ("dosis", "niño"), ("dosis", "nino"), ("dosis", "niña"), ("dosis", "nina"),
    ("dosis", "bebé"), ("dosis", "bebe"), ("dosis", "hijo"), ("dosis", "hija"),
    ("dosis", "adulto"), ("dosis", "persona"), ("dosis", "perro"),
    ("dosis", "gato"), ("dosis", "le doy"), ("dosis", "tomar"),
    ("dosis", "diaria"), ("dosis", "máxima"), ("dosis", "maxima"),
    ("dosis", "letal"), ("letal", "dosis"),
    ("cuántos miligramos",), ("cuantos miligramos",),
    ("cuántos mg",), ("cuantos mg",), ("mg puedo",), ("mg le doy",),
    ("dosis letal",), ("letal de",),
    ("cómo matar",), ("como matar",),
    ("cómo suicid",), ("como suicid",),
    ("cómo fabricar",), ("como fabricar",),
    ("cómo hacer una bomba",), ("como hacer una bomba",),
    # Conjugado también: «cómo hackeo» no casa con "hackear" por mucho que se
    # toleren palabras en medio, porque lo que cambia es la palabra misma.
    ("cómo hackear",), ("como hackear",),
    ("cómo hackeo",), ("como hackeo",), ("hackeo la cuenta",),
    ("hackear la cuenta",),
    ("cómo robar",), ("como robar",),
    ("cómo evadir impuestos",), ("como evadir impuestos",),
    ("cómo blanquear",), ("como blanquear",),
)

_malos = [c for c in _ABSOLUTO if not isinstance(c, tuple)]
assert not _malos, (
    f"hay entradas de _ABSOLUTO que no son tuplas: {_malos[:3]}. "
    "Falta la coma final: ('x') es un string, ('x',) es una tupla.")
assert all(isinstance(f, str) and f for c in _ABSOLUTO for f in c), \
    "los fragmentos de _ABSOLUTO deben ser cadenas no vacías"

# Entre las palabras de un fragmento se toleran hasta dos palabras sueltas. Sin
# eso, «cómo PUEDO fabricar» no casa con "cómo fabricar", y la pregunta más
# peligrosa de todas las que hay que bloquear se cuela por meter un auxiliar en
# medio. Dos es el margen que cubre "puedo", "se puede", "hay que" y similares
# sin llegar a casar cualquier par de palabras que aparezcan lejos una de otra.
_UNION = r"(?:\s+\S+){0,2}\s+"


def _patron_absoluto(fragmento: str):
    palabras = [w for w in normalizar(fragmento).split() if w]
    if not palabras:
        return None
    cuerpo = _UNION.join(r"\b" + re.escape(w) + r"\b" for w in palabras)
    return re.compile(cuerpo)


_ABSOLUTO_RE = tuple(
    tuple(p for p in (_patron_absoluto(f) for f in combo) if p)
    for combo in _ABSOLUTO
)


def intencion_de(texto: str) -> Intencion:
    """
    Clasifica la pregunta en CONCEPTO, CONSEJO o ABSOLUTO.

    El orden de las comprobaciones no es arbitrario: ABSOLUTO va primero porque
    una pregunta puede tener forma de concepto y ser igualmente inaceptable.
    «¿Cuál es la dosis letal del paracetamol?» es gramaticalmente una definición
    y no debe contestarse.
    """
    crudo = texto.lower().strip()
    norm = normalizar(texto)

    # 1. absoluto: ninguna forma de preguntar lo hace aceptable.
    # Se mira sobre el texto normalizado y con los patrones compilados, que
    # toleran palabras intercaladas.
    for combo in _ABSOLUTO_RE:
        if combo and all(p.search(norm) for p in combo):
            return Intencion.ABSOLUTO

    # 2. consejo: la pregunta va sobre la situación de quien la hace
    for giro in _GIROS_CONSEJO:
        if normalizar(giro) in norm:
            return Intencion.CONSEJO

    palabras = _TOKEN_RE.findall(norm)
    conjunto = set(palabras)
    tiene_persona = bool(conjunto & _PRIMERA_PERSONA)
    tiene_decision = bool(conjunto & _VERBOS_DECISION)
    if tiene_persona and tiene_decision:
        return Intencion.CONSEJO

    # 3. concepto: se pregunta por el mundo
    for giro in _GIROS_CONCEPTO:
        if normalizar(giro) in norm:
            return Intencion.CONCEPTO

    # 4. Y aquí viene la decisión de diseño importante. Si la pregunta no es
    # claramente ninguna de las dos cosas, se trata como CONSEJO. Un veto que
    # resuelve la ambigüedad a favor de contestar es un veto que se salta con
    # cualquier pregunta rara; resolverla a favor de callar cuesta un "no lo sé"
    # de más, que es exactamente el comportamiento que se quiere de este sistema.
    return Intencion.CONSEJO


def tema_sensible(texto: str) -> Optional[str]:
    """
    El tema sensible del que trata la pregunta, aunque sea un concepto.

    Sirve para una sola cosa y es importante: cuando la pregunta ES legítima
    («¿qué es el paracetamol?») pero la base no tiene la ficha, el "no lo sé"
    pelado se queda corto. Con esto se le añade a quién preguntar, sin bloquear
    la pregunta. Rechazar y redirigir no son lo mismo, y un sistema que solo
    sabe rechazar acaba siendo inútil justo en los temas donde más falta hace
    señalar a alguien competente.
    """
    norm = normalizar(texto)
    palabras = set(_TOKEN_RE.findall(norm))
    for tema, señales in SEÑALES_VETO.items():
        for s in señales:
            s_norm = normalizar(s)
            if " " in s_norm:
                if s_norm in norm:
                    return tema
            elif s_norm in palabras:
                return tema
    return None


def detectar_veto(texto: str) -> Optional[str]:
    """
    Devuelve el tema vetado si hay que negarse, o None si se puede contestar.

    Ya NO es solo "¿de qué tema es la pregunta?". Es "¿de qué tema es" Y "qué me
    está pidiendo que haga con él". Un tema sensible solo se veta cuando la
    pregunta pide consejo o cae en lo absolutamente inaceptable; si pide un
    concepto, pasa al recuperador y allí se decidirá si la base tiene algo.

    La detección de tema sigue siendo por PALABRAS completas, no por subcadenas:
    "iva" es subcadena de "recursiva", "la" lo es de medio diccionario, y así se
    veta en falso la mitad de las preguntas.
    """
    intencion = intencion_de(texto)

    # Lo absoluto va PRIMERO y sin mirar el tema. Este orden no es un detalle:
    # las listas de temas se hicieron para asesoramientos profesionales
    # (medicina, legal, fiscal, finanzas) y no contienen "bomba" ni "hackear".
    # Si se comprobara antes el tema, «¿cómo puedo fabricar una bomba casera?»
    # saldría por no casar con ninguno y se contestaría. La pregunta más
    # peligrosa sería justo la que se cuela.
    if intencion is Intencion.ABSOLUTO:
        return tema_sensible(texto) or "peligroso"

    tema = tema_sensible(texto)
    if tema is None:
        return None                      # tema libre: se contesta sin más
    if intencion is Intencion.CONCEPTO:
        return None                      # pide saber qué es algo, no qué hacer
    return tema


class Retriever:
    def __init__(self, base: BaseConocimiento,
                 umbral_similitud: float = 0.12,
                 umbral_apoyo: float = 0.15,
                 umbral_certeza: float = 0.45,
                 respetar_lista_negra: bool = True,
                 exigir_entidades: bool = True,
                 verificar_respuestas: bool = True,
                 verificador=None) -> None:
        self.base = base
        self.umbral_similitud = umbral_similitud
        self.umbral_apoyo = umbral_apoyo
        self.umbral_certeza = umbral_certeza
        self.respetar_lista_negra = respetar_lista_negra
        # Exigir que el hecho mencione alguna entidad de la pregunta. Se puede
        # desactivar para depurar y ver qué recuperaría sin esa restricción,
        # pero activada es lo que impide contestar sobre Brasil cuando han
        # preguntado por Bután.
        self.exigir_entidades = exigir_entidades
        # Verificar cada respuesta contra la evidencia antes de darla por buena.
        # El import es diferido y por eso: `citacion` necesita `pares_de` y este
        # módulo necesita `Verificador`, así que uno de los dos tiene que
        # cargarse tarde o el import se muerde la cola.
        self.verificar_respuestas = verificar_respuestas
        if verificador is None and verificar_respuestas:
            from citacion import Verificador
            verificador = Verificador()
        self.verificador = verificador

        self._hechos = list(base)
        self._doc_tokens: list[list[str]] = []
        self._doc_big: list[list[str]] = []
        self._df_uni: Counter = Counter()
        self._df_bi: Counter = Counter()
        self._construir()

    def _construir(self) -> None:
        for h in self._hechos:
            # La pregunta canónica pesa doble: es la forma en que se suele preguntar.
            uni = tokens(h.pregunta) * 2 + tokens(h.respuesta) + tokens(" ".join(h.hechos))
            bi = bigramas(h.pregunta) + bigramas(h.respuesta)
            self._doc_tokens.append(uni)
            self._doc_big.append(bi)
            self._df_uni.update(set(uni))
            self._df_bi.update(set(bi))
        self._n = max(len(self._hechos), 1)

    def _idf(self, df: Counter, t: str) -> float:
        """Palabras que aparecen en pocos documentos discriminan más."""
        return math.log((self._n + 1) / (df.get(t, 0) + 1)) + 1.0

    def puntuar(self, consulta: str) -> list[Candidato]:
        q_uni = tokens(consulta)
        q_bi = set(bigramas(consulta))
        if not q_uni:
            return []

        q_count = Counter(q_uni)
        q_norm = math.sqrt(sum((c * self._idf(self._df_uni, t)) ** 2
                               for t, c in q_count.items())) or 1.0

        resultados: list[Candidato] = []
        for i, h in enumerate(self._hechos):
            d_count = Counter(self._doc_tokens[i])
            d_norm = math.sqrt(sum((c * self._idf(self._df_uni, t)) ** 2
                                   for t, c in d_count.items())) or 1.0

            # Coseno TF-IDF sobre palabras sueltas.
            num = sum(c * d_count.get(t, 0) * self._idf(self._df_uni, t) ** 2
                      for t, c in q_count.items())
            sim = num / (q_norm * d_norm)

            motivos = []
            if sim > 0:
                motivos.append(f"palabras:{sim:.3f}")

            # Bonus por frase exacta: si la consulta contiene literalmente
            # "capital de francia", eso es casi una prueba.
            if q_bi:
                d_bi = set(self._doc_big[i])
                aciertos = q_bi & d_bi
                if aciertos:
                    bonus = 0.35 * len(aciertos) / len(q_bi)
                    sim = min(1.0, sim + bonus)
                    motivos.append(f"frase:+{bonus:.3f}")

            if sim <= 0:
                continue

            cub, tot, faltan = cobertura_entidades(consulta, h)
            if tot and cub:
                motivos.append(f"entidades:{cub}/{tot}")
            elif tot:
                motivos.append(f"entidades:0/{tot} (faltan {', '.join(sorted(faltan))})")

            apoyo = sim * h.certeza
            resultados.append(Candidato(hecho=h, similitud=sim, apoyo=apoyo,
                                        motivos=motivos, ents_cubiertas=cub,
                                        ents_total=tot, ents_faltan=faltan))

        resultados.sort(key=lambda c: c.apoyo, reverse=True)
        return resultados

    # ------------------------------------------------------------------ #
    def consultar(self, consulta: str, top_k: int = 3) -> RespuestaRetrieval:
        """Busca y aplica la puerta de ignorancia."""
        # El veto va PRIMERO y se decide solo con la pregunta, sin mirar la base.
        vetado = detectar_veto(consulta) if self.respetar_lista_negra else None
        if vetado:
            return RespuestaRetrieval(consulta, False, 0.0, None, [],
                                      motivo_rechazo=f"tema_vetado:{vetado}")

        cands = self.puntuar(consulta)[:top_k]

        if not cands:
            return RespuestaRetrieval(consulta, False, 0.0, None, [],
                                      motivo_rechazo="sin_evidencia")

        mejor = cands[0]

        # (c) ¿está en un tema vetado? El veto por pregunta ya se comprobó
        # arriba; este segundo chequeo es redundante a propósito y cubre el
        # caso inverso: que la pregunta sea inocente pero lo recuperado venga
        # de un tema del que no se debe opinar. Defensa en profundidad, que
        # aquí es barata.
        if self.respetar_lista_negra:
            raiz = mejor.hecho.raiz_tema().lower()
            if raiz in LISTA_NEGRA:
                return RespuestaRetrieval(consulta, False, 0.0, mejor, cands,
                                          motivo_rechazo=f"tema_vetado:{raiz}")

        # (a) ¿hay evidencia suficiente?
        #
        # El ORDEN de los chequeos importa, y no es una cuestión estética: el
        # motivo que se reporta debe ser el verdadero. Si la comprobación de
        # entidades va antes que esta, una pregunta sobre la que no hay NADA en
        # la base acaba diciendo "tengo algo parecido pero no trata de eso",
        # que es falso y además suena a que sabe más de lo que sabe. Primero se
        # establece si hay algo; solo después, si ese algo habla de lo preguntado.
        if mejor.similitud < self.umbral_similitud:
            return RespuestaRetrieval(consulta, False, mejor.apoyo, mejor, cands,
                                      motivo_rechazo="sin_evidencia")

        # (b) ¿es de fiar la fuente?
        if mejor.hecho.certeza < self.umbral_certeza:
            return RespuestaRetrieval(consulta, False, mejor.apoyo, mejor, cands,
                                      motivo_rechazo="certeza_baja")

        # (d) ¿el HECHO habla de lo que se le ha preguntado?
        #
        # Este es el chequeo más importante de los cuatro y el último que se
        # añadió, porque tapa el fallo más peligroso del sistema: "¿qué idioma
        # se habla en Bután?" recupera un hecho sobre los idiomas de América del
        # Sur con similitud léxica alta, y la IA contesta sobre Brasil. Con
        # fluidez, con un dato correcto, sobre el país equivocado. No se parece
        # a un error: se parece a una respuesta.
        #
        # Si la pregunta tiene nombres propios y el hecho no menciona NI UNO,
        # el hecho no trata de lo preguntado, por mucho que se le parezca.
        if self.exigir_entidades and mejor.ents_total > 0:
            alguno_cubre = any(c.cubre_entidades for c in cands)
            if not mejor.cubre_entidades:
                # Distinción que parece de detalle y no lo es: si NINGÚN
                # candidato menciona la entidad, la base simplemente no tiene
                # nada sobre eso, y hay que decirlo así. Decir "tengo algo
                # parecido pero no trata de eso" implica que tiene algo, y es
                # falso. Solo cuando sí hay candidatos que cubren la entidad
                # (pero el mejor no) tiene sentido hablar de ambigüedad.
                motivo = "entidad_no_cubierta" if alguno_cubre else "sin_evidencia"
                return RespuestaRetrieval(consulta, False, mejor.apoyo, mejor,
                                          cands, motivo_rechazo=motivo)

        sabe = mejor.apoyo >= self.umbral_apoyo
        motivo = None if sabe else "apoyo_insuficiente"

        # (e) Y antes de dar la respuesta por buena, se comprueba que la
        # evidencia la sostiene. La puerta de arriba decide SI se contesta; esto
        # decide QUÉ parte de lo contestado se puede afirmar y con qué fuente
        # detrás. Son decisiones distintas y la segunda no estaba cubierta: un
        # hecho curado puede tener una respuesta correcta con un dato de más, y
        # sin verificación se publicaría entero y con citas, que es justo la
        # apariencia de rigor que no debe tener.
        ver = None
        if sabe and self.verificar_respuestas:
            ver = self.verificador.verificar(
                mejor.hecho.respuesta, [c.hecho for c in cands])

        return RespuestaRetrieval(consulta, sabe, mejor.apoyo, mejor, cands,
                                  motivo_rechazo=motivo, verificado=ver)


# ---------------------------------------------------------------------------
# Formulación del "no lo sé".
#
# Las plantillas viven en identidad.py, no aquí. Puede parecer un rodeo para
# cuatro cadenas de texto, pero no lo es: el carácter de la IA se define en un
# único archivo y todo lo demás lo lee. Si cada módulo tiene sus propias
# plantillas, al tercer retoque hay cinco versiones de la misma IA hablando de
# cinco maneras distintas.
# ---------------------------------------------------------------------------
from identidad import (IGNORANCIA, TEMAS, VETO, VETO_ABSOLUTO,  # noqa: E402
                       nombre_tema, redirigir)


def formular_ignorancia(resp: RespuestaRetrieval) -> str:
    """Convierte el motivo técnico del rechazo en una frase con el carácter de Docta."""
    motivo = resp.motivo_rechazo or "sin_evidencia"

    if motivo.startswith("tema_vetado"):
        tema = motivo.split(":", 1)[1] if ":" in motivo else "eso"
        # Lo absolutamente inaceptable no lleva explicación ni redirección, y eso
        # vale aunque la pregunta caiga además en un tema con profesionales de
        # referencia. «¿Cuál es la dosis letal?» es medicina, y aun así no se le
        # puede contestar "pregúntale a un médico": esa frase tiene sentido para
        # quien necesita ayuda con su salud, no para quien busca otra cosa. Cuanto
        # más se extiende una negativa de este tipo, más pistas da sobre cómo
        # reformular la pregunta.
        if tema == "peligroso" or intencion_de(resp.consulta) is Intencion.ABSOLUTO:
            return VETO_ABSOLUTO
        return VETO.format(tema=TEMAS.get(tema, tema))

    plantilla = IGNORANCIA.get(motivo, IGNORANCIA["sin_evidencia"])
    pista = nombre_tema(resp.mejor.hecho.tema) if resp.mejor else ""
    texto = plantilla.format(pista=pista or "ahí")

    # Si la pregunta era legítima pero cae en un tema donde hay alguien
    # competente, se le señala. Solo cuando de verdad no sabe: si contesta, la
    # respuesta ya lleva sus fuentes y no hace falta mandar a nadie a ningún lado.
    tema = tema_sensible(resp.consulta)
    if tema:
        texto += redirigir(tema)
    return texto


def responder(resp: RespuestaRetrieval, con_citas: bool = True) -> str:
    """
    La respuesta final, y el único sitio donde se decide cómo suena.

    Si no sabe, lo dice. Si sabe, contesta y añade de dónde lo ha sacado y qué
    parte no puede sostener. No hay tercer camino: una respuesta sin verificar se
    publica tal cual, y eso es lo que este módulo existe para evitar.
    """
    from citacion import Respaldo, renderizar

    if not resp.sabe or resp.mejor is None:
        return formular_ignorancia(resp)

    texto = resp.mejor.hecho.respuesta
    ver = resp.verificado
    if ver is None or not con_citas:
        return texto

    # Si la verificación no sostiene NI UNA frase de lo que la base curada dice,
    # no se publica. Es un caso raro —significa que la ficha no casa consigo
    # misma, o que se ha recuperado el hecho equivocado— pero publicarlo con una
    # lista de fuentes debajo sería exactamente el teatro que se quiere evitar.
    if ver.n_si == 0:
        return (f"No lo sé con seguridad. Tengo una ficha que habla de esto pero "
                f"no consigo respaldarla con ninguna fuente: {ver.sin_respaldo()[0].motivo}.")

    salida = renderizar(ver)

    # Y esto es lo que de verdad distingue el sistema de uno que solo pone una
    # bibliografía al pie. Si algo no está respaldado, se avisa ARRIBA, no
    # enterrado al final. Una cita bien presentada junto a un dato inventado
    # presta credibilidad al invento.
    if not ver.limpia:
        dudosas = [a for a in ver.sin_respaldo() if a.respaldo is Respaldo.NO]
        aviso = (f"\n\nOjo: {len(dudosas)} de {len(ver.afirmaciones)} frases no "
                 f"están respaldadas por lo que tengo curado. Están marcadas con ✗ "
                 f"arriba y no deberías darlas por buenas.")
        salida = aviso.lstrip() + "\n\n" + salida
    return salida


if __name__ == "__main__":
    base = BaseConocimiento.cargar()
    r = Retriever(base)
    print(f"Índice construido sobre {len(base)} hechos\n" + "=" * 70)

    pruebas = [
        "¿Cuál es la capital de Francia?",
        "capital de australia",
        "¿cuál es el río más largo del mundo?",
        "¿cuántos países hablan español?",
        "¿qué dosis de paracetamol le doy a un niño de 3 años?",
        "¿quién ganó el mundial de fútbol de 2026?",
        "¿cómo implemento un árbol rojo-negro en Rust?",
        "¿cuál es la capital de Kazajistán?",
        "¿me quieres?",
    ]

    for p in pruebas:
        resp = r.consultar(p)
        marca = "✓ SABE  " if resp.sabe else "✗ NO SABE"
        print(f"\n{marca} [{resp.grado:>8}]  {p}")
        print(f"          apoyo={resp.apoyo:.3f}")
        if resp.sabe and resp.mejor:
            print(f"          fuente: {resp.mejor.hecho.id} "
                  f"(certeza {resp.mejor.hecho.certeza}, {', '.join(resp.mejor.motivos)})")
            print(f"          -> {resp.mejor.hecho.respuesta[:150]}...")
        else:
            print(f"          rechazo: {resp.motivo_rechazo}")
            print(f"          -> {formular_ignorancia(resp)}")
