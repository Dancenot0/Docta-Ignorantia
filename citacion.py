"""
Respaldo: verificación de afirmaciones y citas.

Tres cosas distintas que conviene no mezclar:

1. ANOTAR. Cada afirmación de la respuesta se compara contra la evidencia
   disponible y se marca como respaldada, dudosa o sin respaldo.

2. CITAR. Lo que está respaldado lleva su referencia. Lo que no, se dice que no.

3. VERIFICAR. Y esto es lo importante: la anotación se le aplica TAMBIÉN al
   texto que genera la red neuronal, no solo al curado. Un modelo de lenguaje
   puede tomar una respuesta correcta de la base y añadirle un dato inventado
   con total fluidez. Sin esta capa, el sistema citaría fuentes mientras
   alucina. Con ella, el dato inventado sale marcado como no respaldado.

Diseño deliberado: la verificación es CONSERVADORA. Ante la duda marca "sin
respaldar", no "respaldado". Un sistema de citas que da por bueno lo que no
puede comprobar es peor que no tener citas, porque añade una apariencia de
rigor que no existe.

Cómo decide si una frase está respaldada:
  * NÚMEROS (la señal fuerte). Extrae las cifras de la frase y exige que cada
    una aparezca en la evidencia, normalizando separadores de miles y decimales.
    Si una frase dice "8.849 metros" y la evidencia no contiene ese número, no
    está respaldada, diga lo que diga el resto. Los datos inventados casi siempre
    son numéricos, y son lo más fácil de pillar.
  * COBERTURA LÉXICA. Qué fracción de las palabras con contenido de la frase
    aparecen en la evidencia. No usa stopwords. Umbral bajo a propósito: esto
    atrapa la invención descarada, no la paráfrasis legítima.
  * BIGRAMAS (la señal que faltaba). La cobertura léxica no puede ver las
    relaciones inventadas: una frase compuesta solo con palabras que SÍ están en
    la fuente pasa como respaldada aunque la relación sea falsa. Ejemplo real del
    test: "Johannesburgo es la sede del parlamento" usa exclusivamente palabras
    de la fuente de Sudáfrica y es falsa —la sede es Ciudad del Cabo. Los pares
    de palabras consecutivas sí la pillan, porque la fuente nunca junta esas
    palabras en ese orden.

Lo que NO hace, y por qué:
  No comprueba si el dato es VERDAD. Comprueba si está RESPALDADO por lo que el
  sistema tiene curado. Son cosas distintas y la distinción es el punto entero:
  una base curada puede estar equivocada. Por eso cada referencia lleva un campo
  `verificado` que solo vale true si un humano abrió la fuente y comprobó el
  dato. Sin ese campo, citar sería teatro.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Optional, Sequence

from knowledge import Hecho, Referencia
from tokenizer import normalizar


class Respaldo(Enum):
    SI = "respaldado"
    PARCIAL = "parcialmente"
    NO = "sin respaldo"


@dataclass
class Afirmacion:
    """Una frase de la respuesta y el veredicto sobre si la evidencia la sostiene."""
    texto: str
    respaldo: Respaldo
    numeros: list[str] = field(default_factory=list)
    numeros_faltan: list[str] = field(default_factory=list)
    cobertura: float = 0.0
    cobertura_bigramas: float = 0.0
    fuente: Optional[Hecho] = None          # hecho que mejor la respalda
    fuentes: list[Hecho] = field(default_factory=list)   # todos los que la respaldan
    motivo: str = ""

    @property
    def marcador(self) -> str:
        return {Respaldo.SI: "✓", Respaldo.PARCIAL: "~", Respaldo.NO: "✗"}[self.respaldo]


@dataclass
class Verificacion:
    texto: str
    afirmaciones: list[Afirmacion]
    evidencia: list[Hecho]

    @property
    def n_si(self) -> int:
        return sum(1 for a in self.afirmaciones if a.respaldo is Respaldo.SI)

    @property
    def n_no(self) -> int:
        return sum(1 for a in self.afirmaciones if a.respaldo is Respaldo.NO)

    @property
    def proporcion_respaldada(self) -> float:
        n = len(self.afirmaciones)
        return (self.n_si + 0.5 * sum(1 for a in self.afirmaciones
                                     if a.respaldo is Respaldo.PARCIAL)) / max(n, 1)

    @property
    def limpia(self) -> bool:
        """¿Se puede publicar tal cual? Solo si no hay NADA sin respaldar."""
        return self.n_no == 0

    @property
    def referencias(self) -> list[Referencia]:
        """
        Referencias de los hechos usados, sin duplicados y en orden.

        Recorre TODOS los hechos que sostienen cada frase, no solo el que mejor
        puntuó. Una respuesta puede apoyarse en dos fichas distintas y citar
        solo una sería citar de menos: el lector se quedaría sin poder comprobar
        una parte de lo que se le está afirmando.
        """
        vistos, salida = set(), []
        for a in self.afirmaciones:
            if a.respaldo is Respaldo.NO:
                continue
            hechos = a.fuentes or ([a.fuente] if a.fuente else [])
            for h in hechos:
                for r in h.referencias:
                    clave = (r.titulo, r.url)
                    if clave not in vistos:
                        vistos.add(clave)
                        salida.append(r)
        return salida

    def sin_respaldo(self) -> list[Afirmacion]:
        return [a for a in self.afirmaciones if a.respaldo is not Respaldo.SI]


# ---------------------------------------------------------------------------
# Extracción de datos
# ---------------------------------------------------------------------------
_STOP = frozenset("""
a al algo algun alguna algunas algunos ante antes aquel aquella aquellas aquello
aquellos aqui asi aun aunque con como contra cual cuales cualquier cuando cuanto
de del ella ellas ello ellos en entre era eran eres es esa esas ese eso esos esta
estaba estan estar estas este esto estos fue fueron ha hace hacia han hasta hay la
las le les lo los mas me mi mis mucho muchos muy nada ni no nos nosotros o os otra
otras otro otros para pero poco por porque que quien quienes se sea ser si sin sino
sobre solo son su sus tambien tan te tiene tienen to toda todas todo todos tu tus un
una unas uno unos usted ustedes va vamos y ya yo ser hay donde cuando como
""".split())

_PALABRA = re.compile(r"[a-záéíóúñü0-9]+")

# Cifras, con separadores de miles y decimales en notación española e inglesa.
_NUMERO = re.compile(r"\d[\d.,]*\d|\d")

# Unidades y magnitudes que suelen acompañar a un dato y ayudan a comparar.
_UNIDADES = ("km", "km²", "m ", "metros", "kilómetros", "millones", "mil",
             "habitantes", "años", "grados", "%", "veces")

_ABREVIATURAS = ("s.", "S.A.", "etc.", "D.C.", "A.C.", "aprox.", "núm.",
                 "pág.", "km.", "n.")

# OJO con cómo se buscan. El primer intento hacía `texto.replace("s.", ...)` a
# pelo y se cargó las frases: "s." es subcadena de "metros.", de "casas.", de
# cualquier palabra acabada en s. Protegía el punto que cerraba la frase y el
# separador dejaba de cortar. Las abreviaturas hay que anclarlas: lo que va
# delante no puede ser letra ni dígito.
_ABREV_RE = {a: re.compile(r"(?<![0-9A-Za-zÁÉÍÓÚÑÜáéíóúñü])" + re.escape(a)
                           + r"(?![0-9A-Za-zÁÉÍÓÚÑÜáéíóúñü])", re.IGNORECASE)
             for a in _ABREVIATURAS}

# Conjunciones que separan proposiciones dentro de una misma frase.
_CONJ = frozenset(("y", "e", "pero", "aunque", "sino", "mas"))

# Formas verbales conjugadas, SIN tildes. Sirven para una sola cosa y es
# importante: saber si una conjunción separa dos proposiciones o solo dos
# elementos de una enumeración. Van sin tildes porque el tokenizer del
# recuperador las quita, y comparar "es" contra "és" no casa nunca.
_VERBO = frozenset("""
es son era eran fue fueron sido ser
esta estan estaba estaban estar estara estaran
tiene tienen tenia tenian tuvo tuvieron tener tendra tendran
ha han habia habian hubo haber hara
hizo hacen hacia hacian hacer
mide miden media median midio medir medira
ocupa ocupan ocupaba ocuparon ocupar
limita limitan
nacio nacieron nace
vive viven vivia vivieron vivir
dice dicen dijo decir
queda quedan quedaba
llama llaman denominaba denomina
compone componen forma forman formaba
fundado fundada situado situada localizado localizada
pertenece pertenecen
alcanza alcanzan
""".split())
_VERBO = frozenset(v for v in _VERBO if v)

# Conjunciones que SIEMPRE separan dos proposiciones, sin más comprobación:
# nadie escribe "pero" o "aunque" para unir dos elementos de una lista.
_CONJ_FUERTE = frozenset(("pero", "aunque", "sino", "mas", "aunque"))

_TOK_PLANO = re.compile(r"[0-9a-z]+")


def _contiene_verbo(texto: str) -> bool:
    """
    ¿Hay un verbo conjugado en este trozo?

    Es la única pieza de "análisis sintáctico" de todo el proyecto y está aquí
    porque hace falta: sin ella no se puede distinguir una enumeración de una
    afirmación nueva. No es un analizador, es una lista de 84 formas verbales, y
    con eso basta para el criterio que se aplica abajo.
    """
    return any(t in _VERBO for t in _TOK_PLANO.findall(normalizar(texto)))


def partir_unidades(texto: str) -> list[str]:
    """
    Parte una frase en unidades de verificación: las proposiciones separadas por
    coma, punto y coma o conjunción.

    Por qué hace falta. La frase corrupta del test es «Sudáfrica tiene tres
    capitales y Johannesburgo es la sede del parlamento». Verificada entera
    puntúa bien, porque la mitad correcta arrastra a la falsa: todas sus palabras
    están en la fuente y buena parte de sus parejas también. Verificada por
    proposiciones, la segunda queda al descubierto —la fuente nunca pone
    "Johannesburgo" junto a "sede del parlamento". Verificar la unidad más
    pequeña que sostiene una afirmación es lo que convierte un detector
    mediocre en uno útil.

    Y la trampa. Cortar por comas a pelo parte los decimales: "44,6 millones"
    se convierte en "Asia tiene unos 44" y "6 millones de kilómetros cuadrados",
    y entonces el verificador busca la cifra 44, que no está en ninguna fuente.
    Es decir, el corte por comas FABRICA un falso positivo sobre un dato que era
    correcto. Las comas entre dígitos se enmascaran antes de cortar y se
    restituyen después.
    """
    enmascarado = re.sub(r"(?<=\d),(?=\d)", "\x02", texto)
    crudo = tokens_crudos(enmascarado)
    if not crudo:
        return [texto]

    cortes = [0]
    for i, (pal, ini, fin) in enumerate(crudo):
        if i == 0:
            continue
        previo = normalizar(crudo[i - 1][0]).lower()
        es_corte = False
        if pal in (",", ";"):
            es_corte = True
        elif previo in ("y", "e"):
            # "y" solo separa proposiciones si hay verbo a los dos lados.
            # «entre Nepal y el Tíbet»: ni a la izquierda ni a la derecha hay
            # verbo, es una enumeración, no se corta.
            # «Sudáfrica tiene tres capitales y Johannesburgo es la sede»:
            # "tiene" a la izquierda, "es" a la derecha, son DOS afirmaciones y
            # cada una se comprueba por su cuenta. Cortar mal por aquí es peor
            # que no cortar: partiría en cachos respuestas perfectamente buenas.
            izquierda = enmascarado[cortes[-1]:ini]
            derecha = enmascarado[fin:]
            if _contiene_verbo(izquierda) and _contiene_verbo(derecha):
                es_corte = True
        elif previo in _CONJ_FUERTE:
            es_corte = True
        if es_corte:
            cortes.append(ini)
    cortes.append(len(enmascarado))

    unidades = []
    for a, b in zip(cortes, cortes[1:]):
        u = enmascarado[a:b].replace("\x02", ",").strip(" ,;")
        if u:
            unidades.append(u)
    return unidades or [texto]


def tokens_crudos(texto: str) -> list[tuple[str, int, int]]:
    """
    Tokens con su posición, para poder cortar el texto original sin perderlo.

    Las comas y los puntos y comas salen como tokens sueltos: son los candidatos
    a punto de corte y hace falta saber dónde están. Las comas entre dígitos ya
    vienen enmascaradas, así que no aparecen aquí.
    """
    return [(m.group(0), m.start(), m.end())
            for m in re.finditer(r"[\wÁÉÍÓÚÑÜáéíóúñü]+|[,;]", texto)]


def partir_frases(texto: str) -> list[str]:
    """
    Parte en frases por puntuación de fin de frase.

    NO se parte por "espacio seguido de mayúscula", que es lo primero que se le
    ocurre a uno y es un error: en español los nombres propios van en mayúscula
    a mitad de frase, así que "El Everest mide 8.849 metros" acabaría partido en
    "El" y "Everest mide...". Verificar cachos de frase no sirve para nada,
    porque un cacho siempre parece respaldado.

    Tampoco se parte por el punto que va entre dígitos ("8.849") ni por el de
    las abreviaturas, que son justo los datos que más interesa comprobar.
    """
    texto = texto.strip()
    if not texto:
        return []

    # protegemos los puntos que NO cierran frase
    protegido = re.sub(r"(?<=\d)\.(?=\d)", "\x00", texto)
    for abrev in _ABREVIATURAS:
        protegido = _ABREV_RE[abrev].sub(
            lambda m, a=abrev: a.replace(".", "\x01"), protegido)

    trozos = re.split(r"(?<=[.!?;:])\s+", protegido)
    salida = []
    for t in trozos:
        t = t.replace("\x00", ".").replace("\x01", ".").strip()
        if len(t) > 1:
            salida.append(t)
    return salida or [texto]


def numeros_de(texto: str) -> list[str]:
    """
    Cifras normalizadas. "8.849" y "8,849" y "8849" deben dar lo mismo, porque
    si no, una fuente que escribe los miles con punto no casará nunca con una
    respuesta que los escribe con espacio.

    Regla de normalización: si el número tiene a la vez . y , se asume notación
    española (punto=miles, coma=decimal). Si solo tiene uno y va seguido de
    exactamente 3 dígitos, se asume que es separador de miles.
    """
    salida = []
    for crudo in _NUMERO.findall(texto):
        n = crudo.strip(".,")
        if not n:
            continue
        tiene_punto, tiene_coma = "." in n, "," in n
        if tiene_punto and tiene_coma:
            if n.rfind(".") < n.rfind(","):        # 8.849,5 -> miles=punto
                n = n.replace(".", "").replace(",", ".")
            else:                                   # 8,849.5 -> miles=coma
                n = n.replace(",", "")
        elif tiene_punto or tiene_coma:
            sep = "." if tiene_punto else ","
            partes = n.split(sep)
            if len(partes) == 2 and len(partes[1]) == 3 and partes[1].isdigit():
                n = "".join(partes)                 # separador de miles
            elif tiene_coma:
                n = n.replace(",", ".")             # decimal español
        try:
            val = float(n)
            salida.append(f"{val:g}")
        except ValueError:
            salida.append(n)
    return salida


def palabras_de(texto: str) -> set[str]:
    """Palabras con contenido, normalizadas y sin stopwords."""
    return {w for w in _PALABRA.findall(normalizar(texto))
            if w not in _STOP and len(w) > 2}


def pares_de(texto: str) -> list[str]:
    """
    Pares de palabras CON CONTENIDO consecutivas, en orden y con sus repeticiones.

    Es la señal que detecta relaciones inventadas, la que las palabras sueltas no
    pueden ver.

    Por qué sin stopwords, y esto costó un fallo de test descubrirlo. La primera
    versión emparejaba todas las palabras y la frase falsa «Johannesburgo es la
    sede del parlamento» pasó de 0 % de parejas a 89 %: los pares "es_la" y
    "la_sede" aparecen en cualquier texto del mundo, así que no discriminan nada.
    Quitar las palabras vacías deja los pares que importan —"johannesburgo_sede",
    "sede_parlamento"— y la fuente de Sudáfrica no los junta nunca.

    Ojo con la tentación de unificar esto con los bigramas del recuperador. Son
    dos trabajos distintos: para PUNTUAR una consulta, los bigramas con
    stopwords afinan el parecido; para VERIFICAR una afirmación, la diluyen.
    """
    ts = [w for w in _PALABRA.findall(normalizar(texto))
          if w not in _STOP and len(w) > 2]
    return [f"{a}_{b}" for a, b in zip(ts, ts[1:])]


# ---------------------------------------------------------------------------
# El verificador
# ---------------------------------------------------------------------------
class Verificador:
    def __init__(self, umbral_cobertura: float = 0.55,
                 tolerancia_numerica: float = 0.005,
                 umbral_bigramas: float = 0.30):
        """
        umbral_cobertura: fracción mínima de palabras con contenido que deben
            aparecer en la evidencia. 0.55 es bajo a propósito: la misión es
            pillar invención descarada, no castigar la paráfrasis.
        tolerancia_numerica: margen relativo para dar un número por presente.
            Cubre redondeos legítimos (44,6 vs 44,58 millones).
        umbral_bigramas: fracción mínima de pares de palabras consecutivas que
            deben aparecer en la evidencia. Es el detector de RELACIONES
            inventadas, que la cobertura léxica no puede ver: véase el
            comentario en `verificar_frase`.
        """
        self.umbral_cobertura = umbral_cobertura
        self.tol = tolerancia_numerica
        self.umbral_bigramas = umbral_bigramas

    # ------------------------------------------------------------------ #
    def _buscar_numero(self, num: str, pool_nums: list[float]) -> bool:
        try:
            v = float(num)
        except ValueError:
            return False
        for c in pool_nums:
            if v == c:
                return True
            escala = max(abs(v), abs(c), 1e-9)
            if abs(v - c) / escala <= self.tol:
                return True
        return False

    def verificar_frase(self, frase: str, hechos: Sequence[Hecho]) -> Afirmacion:
        """
        ¿Está esta frase respaldada por alguno de estos hechos?

        Se comprueba la frase entera Y cada una de sus proposiciones por
        separado, y manda la peor. Sin eso, una frase que pega una afirmación
        falsa detrás de una verdadera sale bien parada: la verdadera aporta las
        palabras y las parejas que la falsa necesita para colarse.
        """
        if not hechos:
            return Afirmacion(frase, Respaldo.NO, motivo="sin evidencia disponible")

        entero = self._puntuar(frase, hechos)

        afirmativas, fuentes = [], []
        for u in partir_unidades(frase):
            res = self._puntuar(u, hechos, sub=u)
            if res.fuente not in fuentes:
                fuentes.append(res.fuente)
            if self._afirma(u):
                afirmativas.append(res)

        # Si la frase se compone de varias proposiciones que afirman algo, cada
        # una tiene que sostenerse por su cuenta. Una sola que no lo haga
        # arrastra a la frase entera: es exactamente el truco que usa un modelo
        # de lenguaje para colar un dato inventado, apoyándolo en otro correcto.
        peor = entero
        if len(afirmativas) >= 2:
            for res in afirmativas:
                if self._orden(res) < self._orden(peor):
                    peor = res
        if peor is not entero:
            peor.texto = frase
            peor.motivo = (f"una de sus proposiciones no está respaldada: "
                           f"{peor.motivo}")
        # las referencias se acumulan de todos los hechos que sostienen algo de
        # la frase, no solo del que mejor puntuó: si cada proposición se apoya en
        # una fuente distinta, citar solo una sería citar de menos.
        peor.fuentes = fuentes
        return peor

    @staticmethod
    def _afirma(unidad: str) -> bool:
        """
        ¿Esta unidad sostiene una afirmación propia?

        Al cortar por comas salen fragmentos que no afirman nada por sí solos:
        «no Sídney ni Melbourne», «en 1851», «y de Asia». Verificarlos como si
        fueran afirmaciones independientes daría "sin respaldo" a frases
        perfectamente correctas, porque un fragmento suelto nunca casa bien con
        la fuente completa. Se exige un verbo conjugado y algo más que una
        palabra con contenido; si no, el fragmento se hereda del veredicto de la
        frase entera.
        """
        return _contiene_verbo(unidad) and len(palabras_de(unidad)) >= 2

    @staticmethod
    def _orden(a: Afirmacion) -> int:
        return {Respaldo.SI: 2, Respaldo.PARCIAL: 1, Respaldo.NO: 0}[a.respaldo]

    def _puntuar(self, texto: str, hechos: Sequence[Hecho],
                 sub: Optional[str] = None) -> Afirmacion:
        """Compara un texto contra cada hecho y devuelve el mejor candidato."""
        nums_frase = numeros_de(texto)
        pal_frase = palabras_de(texto)
        big_frase = pares_de(texto)

        mejor = None
        for h in hechos:
            evidencia = h.texto_completo()
            try:
                pool_floats = [float(n) for n in numeros_de(evidencia)]
            except ValueError:
                pool_floats = []
            pal_evidencia = palabras_de(evidencia)
            big_evid = set(pares_de(evidencia))

            faltan = [n for n in nums_frase if not self._buscar_numero(n, pool_floats)]
            cobertura = (len(pal_frase & pal_evidencia) / len(pal_frase)
                         if pal_frase else 1.0)
            cov_big = (sum(1 for b in big_frase if b in big_evid) / len(big_frase)
                       if big_frase else 1.0)

            if nums_frase and faltan:
                # La frase da una cifra que la evidencia no contiene. Eso es lo
                # más parecido a una prueba directa de invención que hay.
                veredicto = Respaldo.NO
                motivo = f"la cifra {', '.join(faltan)} no aparece en la fuente"
            elif cobertura < self.umbral_cobertura * 0.6:
                veredicto = Respaldo.NO
                motivo = f"no casa con la fuente (cobertura {cobertura:.0%})"
            elif cov_big < self.umbral_bigramas:
                # El caso que la cobertura léxica NO puede ver, y que justifica
                # este segundo detector. "Johannesburgo es la sede del
                # parlamento" usa exclusivamente palabras que están en la fuente
                # de Sudáfrica —cobertura del 100 %— y es falsa: la sede del
                # parlamento es Ciudad del Cabo. Contar palabras no distingue
                # "todas las piezas correctas" de "las piezas correctas mal
                # combinadas". Las parejas consecutivas sí: una relación
                # inventada produce pares que la fuente nunca junta.
                veredicto = Respaldo.NO
                motivo = (f"las palabras están en la fuente pero la relación no "
                          f"(solo {cov_big:.0%} de las parejas consecutivas)")
            elif cobertura < self.umbral_cobertura:
                veredicto = Respaldo.PARCIAL
                motivo = f"cobertura léxica baja ({cobertura:.0%})"
            else:
                veredicto = Respaldo.SI
                motivo = (f"cobertura léxica {cobertura:.0%}, "
                          f"{cov_big:.0%} de parejas"
                          + (", cifras comprobadas" if nums_frase else ""))

            clave = (self._orden(Afirmacion("", veredicto)),
                     cobertura + cov_big, -len(faltan))
            if mejor is None or clave > mejor[0]:
                mejor = (clave, Afirmacion(
                    texto=sub or texto, respaldo=veredicto, numeros=nums_frase,
                    numeros_faltan=faltan, cobertura=cobertura,
                    cobertura_bigramas=cov_big, fuente=h, motivo=motivo))

        return mejor[1]

    def verificar(self, texto: str, evidencia: Iterable[Hecho]) -> Verificacion:
        """Verifica un texto completo contra un conjunto de hechos."""
        hechos = list(evidencia)
        afirmaciones = [self.verificar_frase(f, hechos) for f in partir_frases(texto)]
        return Verificacion(texto=texto, afirmaciones=afirmaciones, evidencia=hechos)


# ---------------------------------------------------------------------------
# Renderizado con citas
# ---------------------------------------------------------------------------
SIN_FUENTE = "sin fuente documentada"


def renderizar(ver: Verificacion, inline: bool = True,
               listar_referencias: bool = True, mostrar_dudas: bool = True) -> str:
    """
    Compone la respuesta final.

    Con `inline`, cada frase lleva su marca. Es más feo de leer y más honesto:
    quien recibe la respuesta ve en el momento qué parte está sostenida y qué
    parte no, sin tener que pedirlo.
    """
    lineas = []
    for a in ver.afirmaciones:
        if inline:
            lineas.append(f"{a.marcador} {a.texto}")
        else:
            lineas.append(a.texto)

    salida = "\n".join(lineas)

    dudas = ver.sin_respaldo()
    if mostrar_dudas and dudas:
        salida += "\n\n—\n"
        for a in dudas:
            if a.respaldo is Respaldo.NO:
                salida += f"✗ No puedo respaldar: «{a.texto}» ({a.motivo}).\n"
            else:
                salida += f"~ Solo en parte: «{a.texto}» ({a.motivo}).\n"
        salida = salida.rstrip()

    if listar_referencias:
        refs = ver.referencias
        if refs:
            salida += "\n\nFuentes:\n"
            for i, r in enumerate(refs, 1):
                marca = "comprobada" if r.verificado else "SIN COMPROBAR"
                linea = f"  [{i}] {r.cita_corta()} — {marca}"
                if r.url:
                    linea += f"\n      {r.url}"
                if r.nota:
                    linea += f"\n      {r.nota}"
                salida += linea + "\n"
            salida = salida.rstrip()
        else:
            # Y aquí está la parte que de verdad importa. Si no hay fuentes, se
            # dice. Un sistema que solo lista las fuentes que tiene acaba
            # pareciendo que respalda todo cuando en realidad no respalda nada.
            salida += f"\n\nFuentes: {SIN_FUENTE}. Esto viene de la base curada " \
                      f"pero nadie ha documentado de dónde se sacó."
    return salida


def resumir(ver: Verificacion) -> str:
    """Una línea de estado, para logs y para la interfaz."""
    n = len(ver.afirmaciones)
    return (f"{ver.n_si}/{n} frases respaldadas"
            f"{'' if ver.limpia else f' · {ver.n_no} sin respaldo'}"
            f" · fuentes: {len(ver.referencias)}")
