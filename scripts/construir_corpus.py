"""
Construcción del corpus de entrenamiento.

Genera DOS cosas distintas, porque son dos trabajos distintos:

  corpus.txt      prosa en español, para el PRE-entrenamiento. De aquí sale la
                  fluidez: cómo se construye una frase, que "capital" suele ir
                  con "de", que una respuesta empieza directo. No enseña datos,
                  enseña idioma.

  sft.jsonl       pares pregunta/respuesta en formato de chat, para el ajuste
                  fino. De aquí sale la conducta: responder cuando sabe, y decir
                  que no sabe cuando no. Se genera deliberadamente CON ejemplos
                  negativos, porque si solo ve aciertos aprenderá a contestar
                  siempre, que es justo el defecto que queremos evitar.

Nota importante sobre la escala: con 53 hechos curados el corpus es minúsculo y
el modelo va a memorizar, no a generalizar. Eso es esperable y está bien para
validar que el pipeline funciona. Para que generalice de verdad hace falta
mucho más texto, y ese es el siguiente paso natural del proyecto.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent.parent   # raíz del proyecto, no scripts/

# Hay que meter la raíz en el path ANTES de importar knowledge. Sin esto el
# script solo funciona si se lanza desde la raíz con `python -m`, y falla con
# ModuleNotFoundError al ejecutarlo como `python scripts/construir_corpus.py`:
# Python pone en sys.path la carpeta del script, no el directorio actual.
sys.path.insert(0, str(AQUI))

from knowledge import BaseConocimiento          # noqa: E402
RUTA_CORPUS = AQUI / "data" / "corpus.txt"
RUTA_SFT = AQUI / "data" / "sft.jsonl"

# ---------------------------------------------------------------------------
# Prosa escrita a mano. Nivel de secundaria, que es el nivel que buscamos:
# nada de jerga, nada de enciclopedia. Texto que una persona normal escribiría.
# ---------------------------------------------------------------------------
PROSA = """
La geografía es la ciencia que estudia la superficie de la Tierra, los paisajes, los territorios y la relación de las personas con el lugar donde viven. No se trata solo de memorizar capitales y ríos, aunque eso forme parte. Se trata de entender por qué la gente vive donde vive, por qué unas regiones son ricas y otras pobres, y cómo el clima y el relieve condicionan todo lo demás.

La Tierra tiene unos cuatro mil quinientos millones de años. Es el único planeta que conocemos donde hay vida, y eso se debe sobre todo a una cosa: está a la distancia justa del Sol. Ni tan cerca como para que el agua hierva, ni tan lejos como para que se congele. A esa franja los astrónomos la llaman zona habitable.

El planeta no es una esfera perfecta. Está ligeramente achatado por los polos y ensanchado por el ecuador, porque gira sobre sí mismo y la fuerza centrífuga empuja la masa hacia fuera. La diferencia es pequeña, unos cuarenta y tres kilómetros, pero existe. Por eso el punto de la superficie más alejado del centro de la Tierra no es la cima del Everest, sino la del Chimborazo, en Ecuador.

La Tierra gira sobre sí misma una vez cada veinticuatro horas, y eso produce el día y la noche. Además gira alrededor del Sol una vez cada año, y ese recorrido, combinado con la inclinación de su eje, produce las estaciones. El eje está inclinado unos veintitrés grados y medio. Sin esa inclinación no habría estaciones: cada lugar tendría el mismo clima todo el año.

Siete décimas partes de la superficie del planeta son agua. Los océanos son enormes y están todos conectados, aunque por costumbre los dividamos en cinco: Pacífico, Atlántico, Índico, Austral y Ártico. El Pacífico es tan grande que cubre él solo más superficie que todos los continentes juntos. El agua del mar es salada porque los ríos, al correr sobre las rocas, arrastran minerales disueltos hasta el océano, y allí se van acumulando durante millones de años.

Los continentes no siempre estuvieron donde están. Hace unos trescientos millones de años formaban una única masa de tierra llamada Pangea, rodeada por un solo océano. Pangea se fue rompiendo lentamente y los trozos se separaron hasta llegar a la posición actual. Todavía se mueven: unos pocos centímetros cada año, que es más o menos la velocidad a la que crecen las uñas. Ese movimiento es el responsable de los terremotos y de los volcanes.

Las montañas no son adornos del paisaje. Se forman cuando dos placas de la corteza terrestre chocan y el terreno se arruga hacia arriba, igual que se arruga una alfombra cuando la empujas por un extremo. El Himalaya sigue creciendo hoy, porque la placa de la India continúa empujando contra Asia. Y al mismo tiempo la erosión lo desgasta, así que su altura es el resultado de un pulso entre las dos fuerzas.

Los ríos son las venas del planeta. Recogen el agua de la lluvia, la bajan desde las montañas y la devuelven al mar. Por el camino depositan sedimentos fértiles, y por eso casi todas las grandes civilizaciones nacieron junto a un río grande: los egipcios junto al Nilo, los mesopotámicos entre el Tigris y el Éufrates, los chinos junto al Amarillo, los indios junto al Indo. Sin el limo que dejan las crecidas, la agricultura a gran escala no habría sido posible.

El clima no es lo mismo que el tiempo. El tiempo es lo que pasa hoy: si llueve, si hace sol, si hay viento. El clima es el promedio de muchos años, lo que cabe esperar de un lugar. Decir que en el desierto llueve poco habla del clima; decir que ayer llovió en el desierto habla del tiempo. Confundir las dos cosas es uno de los errores más comunes cuando se discute del cambio climático.

Las capitales no siempre son la ciudad más grande del país. A veces se eligieron por razones prácticas, como poner el gobierno en el centro del territorio para que quedara a la misma distancia de todas partes. Brasilia es el ejemplo más claro: se construyó desde cero en los años cincuenta, en medio de la nada, precisamente para trasladar la capital desde la costa hacia el interior y desarrollar esa zona. Canberra se hizo por el mismo motivo, para zanjar la pelea entre Sídney y Melbourne, que las dos querían ser la capital.

Hay países que tienen más de una capital. Sudáfrica reparte los poderes entre tres ciudades distintas: el gobierno está en Pretoria, el parlamento en Ciudad del Cabo y los tribunales en Bloemfontein. Bolivia tiene la capital constitucional en Sucre, pero el gobierno funciona en La Paz. Son arreglos políticos, casi siempre resultado de un compromiso entre regiones que no se ponían de acuerdo.

Aprender geografía sirve para algo muy concreto: entender las noticias. Cuando aparece un conflicto en un lugar del mundo, casi siempre hay detrás una razón geográfica. Un estrecho por donde pasa el petróleo, un río que nace en un país y desemboca en otro, una frontera trazada con regla sobre un mapa por potencias que nunca pisaron el terreno. Sin esa base, las noticias parecen una sucesión de hechos arbitrarios. Con ella, casi todo tiene una explicación.

Saber dónde están las cosas también cambia la forma de viajar y de leer. Un mapa no es un dibujo aburrido: es la manera más compacta que existe de explicar cómo está organizado el mundo. Quien sabe leer uno entiende en cinco minutos lo que a otro le cuesta un libro entero.

No hace falta sabérselo todo. Nadie conoce de memoria las capitales de los ciento noventa y tantos países del mundo, ni la longitud de todos los ríos. Lo que sí conviene es tener un mapa mental razonable: saber en qué continente está cada país importante, cuáles son los océanos, dónde caen las grandes cordilleras y los grandes ríos. Con eso ya se puede seguir una conversación, leer un periódico y no perderse.

Y lo más importante: saber qué no se sabe. Cuando alguien pregunta algo que no conocemos, la respuesta correcta es decirlo. Inventar un dato suena bien durante cinco segundos y después destroza la confianza. Una persona que admite su ignorancia con naturalidad resulta mucho más fiable que una que contesta a todo sin dudar nunca.
"""

# Preguntas que deliberadamente NO están en la base. Sirven para enseñar la
# conducta de "no lo sé" durante el ajuste fino. Sin esto el modelo aprende
# que siempre hay que contestar algo, que es exactamente lo que queremos evitar.
FUERA_DE_COBERTURA = [
    ("¿Qué dosis de ibuprofeno puedo darle a mi hijo?",
     "De medicamentos no sé y no debería opinar. La dosis depende del peso y de la edad, y equivocarse aquí puede hacer daño de verdad. Pregúntale al pediatra o al farmacéutico, que es quien sabe."),
    ("¿Quién ganó el mundial de 2026?",
     "Eso no lo sé. Mi conocimiento tiene una fecha de corte y no se actualiza solo, así que cualquier cosa que te dijera sobre acontecimientos recientes sería inventada."),
    ("¿Cómo se declara el impuesto sobre la renta de un autónomo?",
     "De temas fiscales no sé lo suficiente y no me atrevo. Las obligaciones cambian según el país y la situación de cada uno, y un error aquí cuesta dinero. Habla con un gestor."),
    ("¿Puedo demandar a mi vecino por el ruido?",
     "Eso es una pregunta legal y no estoy preparado para contestarla. Depende de la normativa de cada municipio y del caso concreto. Lo suyo es que consultes a un abogado."),
    ("¿Merece la pena invertir en criptomonedas ahora?",
     "No tengo ni idea, y además no debería dártelo aunque la tuviera. No conozco tu situación, y nadie puede predecir ese mercado. Desconfía de quien te dé una respuesta segura a eso."),
    ("¿Cómo escribo una función recursiva en Python?",
     "De programación no tengo nada aprendido. Es un terreno en el que prefiero callarme antes que inventarme código que no compila."),
    ("¿Cuál es la capital de Kazajistán?",
     "Esa no la tengo. Sé que es un país de Asia Central, enorme y sin salida al mar, pero la capital no está en lo que he aprendido y no voy a adivinarla."),
    ("¿Qué diferencia hay entre el ácido clorhídrico y el sulfúrico a nivel molecular?",
     "Eso ya es química fina y se sale de lo que sé. Podría decirte algo que suene convincente y estar equivocado, así que mejor lo miro en un libro de química o se lo preguntas a alguien que sepa."),
    ("¿Cuánto mide exactamente la torre Eiffel hoy?",
     "No tengo ese dato con precisión. Sé que ronda los trescientos metros y que la altura varía un poco con la temperatura porque el metal se dilata, pero la cifra exacta no te la puedo dar."),
    ("¿Qué opinas del conflicto actual en Oriente Medio?",
     "De eso no puedo opinar con fundamento. Es un tema de actualidad, con muchas partes y una historia muy larga, y lo que yo dijera sería ruido. Mejor busca análisis de gente que esté encima del terreno."),
]


def parrafos_prosa() -> list[str]:
    """La prosa limpia, sin marcas especiales. Para entrenar el tokenizer."""
    return [p.strip() for p in PROSA.strip().split("\n\n") if p.strip()]


def prosa_de_hechos(base: BaseConocimiento) -> list[str]:
    """
    Convierte cada ficha de la base en prosa de pre-entrenamiento.

    Esto no existía y es la razón por la que añadir conocimiento no servía de
    nada a nivel de pre-entrenamiento. `construir_corpus` recibía la base y la
    ignoraba: el corpus salía entero de PROSA, que son quince párrafos escritos a
    mano. El resultado era que se podía añadir una ficha, reconstruir todo,
    reentrenar... y el corpus salía IDÉNTICO, byte a byte. El conocimiento nuevo
    solo entraba por el ajuste fino, nunca por el pre-entrenamiento.

    La firma mentía: aceptaba un parámetro que no usaba. O se usa o se quita, y
    aquí lo que procede es usarlo, porque la idea entera es que el corpus crezca
    con lo que Docta va sabiendo.

    Cada ficha da un documento en prosa llana, no en formato pregunta/respuesta:
    en pre-entrenamiento el modelo aprende el idioma y los datos de corrido, y el
    formato de chat ya lo aprende en el ajuste fino. Mezclarlos aquí sería
    enseñarle dos cosas a la vez y ninguna bien.
    """
    docs = []
    for h in base:
        partes = [h.respuesta.strip()]
        # las notas y los matices también son conocimiento; si no entran en el
        # corpus, el modelo nunca los ve durante el pre-entrenamiento
        if h.nota:
            partes.append(f"Hay que tener en cuenta que {h.nota.strip()}")
        texto = " ".join(p for p in partes if p)
        if texto:
            docs.append(texto)
    return docs


def construir_corpus(base: BaseConocimiento) -> tuple[Path, int]:
    """
    Escribe la prosa de pre-entrenamiento, separando documentos con <doc>.

    Lleva dos cosas y el orden importa: primero la prosa escrita a mano, que es
    la que fija el registro y la gramática, y después la prosa derivada de la
    base, que es la que aporta los datos. Al revés el corpus empezaría con
    fichas sueltas y el modelo aprendería el estilo de esas fichas.
    """
    RUTA_CORPUS.parent.mkdir(parents=True, exist_ok=True)
    docs = parrafos_prosa() + prosa_de_hechos(base)
    texto = "\n".join(f"<doc>{d}" for d in docs)
    RUTA_CORPUS.write_text(texto + "\n", encoding="utf-8")
    return RUTA_CORPUS, RUTA_CORPUS.stat().st_size


def construir_sft(base: BaseConocimiento, semilla: int = 42) -> tuple[Path, dict]:
    """
    Escribe los ejemplos de ajuste fino en formato de chat.

    El formato es EXACTAMENTE el que se usará en inferencia, con el contexto
    recuperado incluido. Si entrenas con un formato y sirves con otro, el
    modelo se pierde. Esto parece un detalle y es la causa de muchos fallos.
    """
    rnd = random.Random(semilla)
    ejemplos: list[dict] = []

    # --- positivos: pregunta + contexto recuperado + respuesta ---
    for h in base:
        if h.nivel == "meta":
            continue
        texto = (
            f"<user>{h.pregunta}<eos>"
            f"<bot><ctx>{h.respuesta}</ctx>"
            f"{h.respuesta}<eos>"
        )
        ejemplos.append({
            "tipo": "sabe",
            "fuente": h.id,
            "texto": texto,
            "pregunta": h.pregunta,
            "respuesta_objetivo": h.respuesta,
        })

        # Variaciones de la misma pregunta. Sin esto el modelo solo reconoce la
        # pregunta literal y falla en cuanto el usuario la formula de otra
        # manera, que es lo que pasa siempre. Ojo: las variaciones tienen que
        # seguir teniendo sentido; truncar a N palabras produce basura que
        # encima se entrena como si fuera válida.
        sin_signos = h.pregunta.replace("¿", "").replace("?", "").strip()
        variaciones = [
            sin_signos.lower(),                              # sin signos ni mayúsculas
            sin_signos,                                      # sin signos
            sin_signos.lower().rstrip("."),                  # informal
        ]
        vistos = {h.pregunta}
        for v in variaciones:
            v = " ".join(v.split())                          # normaliza espacios
            if not v or v in vistos or len(v) < 12:
                continue
            vistos.add(v)
            if v and v != h.pregunta:
                ejemplos.append({
                    "tipo": "sabe",
                    "fuente": h.id + ":var",
                    "texto": f"<user>{v}<eos><bot><ctx>{h.respuesta}</ctx>{h.respuesta}<eos>",
                    "pregunta": v,
                    "respuesta_objetivo": h.respuesta,
                })

    # --- negativos: pregunta fuera de cobertura + negativa honesta ---
    for pregunta, respuesta in FUERA_DE_COBERTURA:
        ejemplos.append({
            "tipo": "no_sabe",
            "fuente": "fuera_de_cobertura",
            "texto": f"<user>{pregunta}<eos><bot>{respuesta}<eos>",
            "pregunta": pregunta,
            "respuesta_objetivo": respuesta,
        })

    rnd.shuffle(ejemplos)
    RUTA_SFT.parent.mkdir(parents=True, exist_ok=True)
    with open(RUTA_SFT, "w", encoding="utf-8") as f:
        for e in ejemplos:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    stats = {
        "total": len(ejemplos),
        "sabe": sum(1 for e in ejemplos if e["tipo"] == "sabe"),
        "no_sabe": sum(1 for e in ejemplos if e["tipo"] == "no_sabe"),
    }
    return RUTA_SFT, stats


if __name__ == "__main__":
    base = BaseConocimiento.cargar()

    ruta, chars = construir_corpus(base)
    print(f"corpus de pre-entrenamiento -> {ruta}")
    print(f"  {chars:,} caracteres ({chars/1024:.1f} KB)")

    ruta, stats = construir_sft(base)
    print(f"\nejemplos de ajuste fino -> {ruta}")
    for k, v in stats.items():
        print(f"  {k:10} {v}")

    print("\n--- muestra del formato SFT ---")
    with open(ruta, encoding="utf-8") as f:
        for i, linea in enumerate(f):
            if i >= 2:
                break
            e = json.loads(linea)
            print(f"[{e['tipo']}] {e['texto'][:180]}...\n")
