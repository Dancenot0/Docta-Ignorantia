# Docta

### Una IA desde cero, con conocimientos de humano promedio

**Docta**, por *docta ignorantia* (Nicolás de Cusa, 1440): cuanto más sabe
alguien, más exactamente conoce los límites de lo que sabe, y por tanto la
ignorancia reconocida con precisión es la forma más alta de conocimiento.

Eso no es un adorno filosófico. Es la especificación técnica del sistema: cada
hecho curado lleva una cifra de certeza, la puerta mide si hay evidencia
suficiente, y el grado resultante decide el tono de la respuesta. **Saber qué no
se sabe está implementado, no declarado.**

Carácter: **sobrio**. Frases cortas, el dato primero, sin disculparse por
existir. No es una elección estética: un sistema que admite ignorancia
constantemente no puede permitirse ser efusivo, porque la efusividad suena
falsa justo al lado de un "no lo sé".

```
Docta 0.1 · tiny    (7,7M)    ← el actual
Docta 0.1 · toy     (16M)     mismo temario, motor más grande
Docta 0.2 · small   (95M)     cuando haya corpus de verdad
Docta 0.3 · medium  (1.3B)    con GPU alquilada
```

La versión sube cuando cambia **el temario o la arquitectura**, no el tamaño.

---

No es un modelo grande. Es **la arquitectura de un modelo grande con tamaño de
juguete**, y encima una capa de conocimiento externo. La premisa del proyecto es
que una IA que lo sepa todo no sirve para nada: lo que sirve es una que sepa lo
que sabe una persona normal y que **diga que no lo sabe** cuando se sale de ahí.

---

## La decisión de diseño que lo explica todo

Con 2 CPUs, 1 GB de RAM y sin GPU no se entrena nada grande. Así que el
conocimiento **no vive en los pesos**: vive en una base curada en texto plano, y
el modelo neural pone el lenguaje y el razonamiento.

```
                 pregunta del usuario
                          │
                          ▼
        ┌──────────────────────────────────┐
        │  1. RETRIEVAL sobre la base      │  ← aquí vive el conocimiento
        │     curada (TF-IDF + frases)     │
        └──────────────────────────────────┘
                          │
                          ▼
        ┌──────────────────────────────────┐
        │  2. PUERTA DE IGNORANCIA         │  ← lo que hace que no alucine
        │  ¿hay evidencia? ¿es fiable?     │
        │  ¿el tema está permitido?        │
        └──────────────────────────────────┘
              │                      │
          SÍ, sabe                NO sabe
              │                      │
              ▼                      ▼
   ┌────────────────────┐   ┌──────────────────┐
   │ 3. El modelo neural│   │ "No lo sé, y por │
   │ redacta la         │   │  qué" + a dónde  │
   │ respuesta CON el   │   │  mirar           │
   │ contexto inyectado │   └──────────────────┘
   └────────────────────┘
              │
              ▼
       ┌─────────────┐
       │  4. MEMORIA  │  conversaciones y correcciones
       └─────────────┘
              │
              ▼
       ┌─────────────┐
       │  5. SUEÑOS   │  replay + recombinación + reentrenamiento
       └─────────────┘     de la política de exploración (Dream-RSI)
```

Un cerebro chiquito con buena memoria externa se comporta como una persona
cultísima en los temas que tiene curados y honestamente ignorante en el resto.
Eso es justo el objetivo.

---

## Qué hay construido y en qué estado

| archivo | qué es | estado |
|---|---|---|
| `config.py` | Arquitectura por presets. Misma estructura, distinto tamaño. | ✅ |
| `tokenizer.py` | BPE a nivel de byte, escrito desde cero. Sin `tiktoken`. | ✅ entrenado, ida y vuelta perfecta |
| `model.py` | Transformer estilo DeepSeek: RoPE, RMSNorm, GQA, SwiGLU, MoE con balanceo sin loss auxiliar. **Forward y backward a mano en NumPy.** | ✅ **gradientes verificados** |
| `knowledge.py` | Esquema de la base de conocimiento + carga. Incluye `Referencia` y el campo `verificado`. | ✅ 53 hechos |
| `citacion.py` | **Verificación frase a frase y citas.** Cifras, cobertura léxica y pares consecutivos. | ✅ 13/13 tests |
| `retriever.py` | TF-IDF + cobertura de entidades + puerta de ignorancia + verificación antes de afirmar. | ✅ 46/46 tests |
| `identidad.py` | Nombre, carácter, tonos y respuestas de ignorancia. Un solo sitio. | ✅ |
| `docta.py` | Punto de entrada: conversación por terminal. | ✅ |
| `tests/test_puerta.py` | Falsos negativos Y falsos positivos del filtro. | ✅ 46/46 |
| `tests/test_citacion.py` | 6 respuestas correctas deben pasar y 7 corruptas deben pillarse. | ✅ 13/13 |
| `scripts/construir_kaggle.py` | Empaqueta código y conocimiento por separado. | ✅ probado |
| `scripts/versionar_conocimiento.py` | Manifiesto y versión de la base. | ✅ |
| `temario.py` | Qué debe saber, sacado de los dos PDF, con informe de cobertura. | ✅ 32 % |
| `dataset.py` | Lotes de pre-entrenamiento y de SFT con máscara. | ✅ |
| `scripts/check_grad.py` | Comprobación numérica de gradientes. | ✅ CORRECTO |
| `scripts/entrenar.py` | Bucle AdamW + schedule + evaluación. | ✅ entrena |
| `scripts/construir_corpus.py` | Genera corpus y SFT desde la base curada. | ✅ |
| `kaggle/docta_kaggle.ipynb` | El proyecto entero en una celda, listo para Kaggle. | ✅ 202 KB |
| memoria episódica | aprender de la conversación | ⬜ |
| bucle de sueños | Dream-RSI | ⬜ |
| generación / chat | muestreo y bucle conversacional | ⬜ |

### Verificación de gradientes

Un backward escrito a mano puede fallar **en silencio**: la pérdida baja igual,
solo que hacia un sitio peor. Así que no vale con que entrene, hay que comprobar
que las derivadas son las derivadas. Se perturban los pesos y se compara con la
definición de derivada.

```
MoE      diff. absoluta máxima 3.6e-07   (escala de gradientes 1.6e-01)   CORRECTO
denso    diff. absoluta máxima 7.0e-10                                     CORRECTO
```

El router del MoE se excluye del criterio y no es un apaño: el top-k es una
operación **discreta**, así que al perturbar un peso un token puede cambiar de
experto y la pérdida pega un salto. La derivada numérica a través de un punto de
discontinuidad no significa nada. Se distingue de un bug porque el error es de
unos pocos por ciento con el signo correcto; un bug de verdad da error ~1.0 o el
signo opuesto.

### Entrenamiento de humo

`tiny`, 7,7M parámetros, 200 pasos, 184 s en 2 CPUs sin GPU:

```
época   1   ent=10.94   val=7.97   acierto= 0.0 %
época   5   ent= 3.30   val=3.14   acierto=51.1 %
época  20   ent= 1.77   val=1.95   acierto=61.7 %
época  40   ent= 0.18   val=0.71   acierto=89.8 %
```

⚠ **Ese 0.71 de validación no demuestra generalización.** El corpus son 2.264
tokens y el modelo tiene 7,7 millones de parámetros: la "validación" son las
mismas frases con ventanas distintas. Está memorizando, que es lo correcto y lo
esperable con esta cantidad de datos. Sirve para probar que el bucle funciona;
para que generalice hace falta más texto, no más épocas.

---

## Los tres bucles de conocimiento

**1. Curado.** Hechos en `data/conocimiento/*.jsonl`, escritos a mano. Cada uno
lleva `certeza` (0–1), `nivel`, `fuente` y los hechos atómicos que lo componen.
Es auditable: abres el archivo y ves lo que sabe. Se corrige sin reentrenar.

**2. Aprende hablando.** Cada conversación va a una memoria episódica; las
correcciones del usuario se guardan con el dato anterior, el nuevo y el motivo,
y alimentan el reentrenamiento. *(pendiente)*

**3. Sueña.** Al estilo Dream-RSI, y con un matiz importante: **no** se
reescriben los pesos del modelo base. Lo que se auto-mejora recursivamente es la
*política de exploración*. El agente graba su árbol de descubrimientos durante
una tarea real y después, en un rato libre, recorre y recombina ese historial
como si fuera un simulador cerrado: puede evaluar miles de estrategias
alternativas a coste de cómputo cero, en vez de volver a ejecutar pruebas caras
desde cero. *(pendiente)*

---

## Cómo se usa

```bash
cd ia
python3 -m venv .pydeps && .pydeps/bin/pip install numpy

.pydeps/bin/python docta.py                          # conversar con Docta

.pydeps/bin/python config.py                         # tamaños y coste en RAM
.pydeps/bin/python scripts/construir_corpus.py      # genera corpus.txt y sft.jsonl
.pydeps/bin/python scripts/entrenar_tokenizer.py    # entrena el BPE
.pydeps/bin/python scripts/check_grad.py            # VERIFICA el backward
.pydeps/bin/python scripts/entrenar.py --modo pre --preset tiny --epocas 40
.pydeps/bin/python tests/test_puerta.py              # 46 comprobaciones
.pydeps/bin/python tests/test_citacion.py            # 13 casos de verificación
.pydeps/bin/python scripts/construir_kaggle.py       # regenera el notebook
.pydeps/bin/python identidad.py                      # nombre, tonos, negativas
```

---

## Ejecutarlo en Kaggle

```bash
.pydeps/bin/python scripts/construir_kaggle.py
```

Genera tres cosas en `kaggle/`:

| archivo | qué es |
|---|---|
| `docta_kaggle.ipynb` | El que se sube a Kaggle. 222 KB, autocontenido. |
| `docta_instalador.py` | Lo mismo como script: `python docta_instalador.py`. |
| `docta_dataset_kaggle.zip` | Plan B: el proyecto sin comprimir-en-celda, para subirlo como Dataset. |

**El notebook es autocontenido.** El proyecto entero viaja comprimido y en
base64 dentro de las celdas marcadas `PAQUETE`: 132 KB de `.tar.gz`, 178 KB
codificados. Son 25 archivos: 9 módulos, 5 scripts, 2 suites de tests y 9 de
datos, con las 53 fichas curadas, el corpus y el tokenizer. Al ejecutarse, la
celda `DEPLIEGUE` lo monta en `./docta` y ya.

Se eligió así frente a las dos alternativas porque las dos añaden fricción donde
no hace falta. Subirlo como Dataset de Kaggle obliga a tener cuenta, a mantener
el dataset sincronizado y a que quien quiera probarlo repita los pasos. Pegar
5.000 líneas en celdas es ilegible y no se puede mantener. Que quepa en el
notebook resuelve las dos cosas a la vez.

### Qué no viaja y por qué

Los **checkpoints** (28 MB). No caben en una celda y sobran: el motivo de ir a
Kaggle es entrenar allí, no llevarse los pesos entrenados. La celda 5 los
regenera.

### Las celdas

| celda | qué hace | tarda |
|---|---|---|
| `PAQUETE` ×12 | Llevan el proyecto comprimido. **No se leen ni se tocan.** | — |
| `DEPLIEGUE` | Lo monta en `./docta` y comprueba que carga entero | segundos |
| `PUERTA` | 9 preguntas: sabe, duda, calla y veta | segundos |
| `TESTS` | Las dos suites: 46 + 13 casos | segundos |
| `GRADIENTE` | Backprop contra derivada numérica | ~1 min |
| `ENTRENAR` | **Pre-entrenamiento.** ~2 min en Kaggle con `tiny` | minutos |
| `PREGUNTAR` | Hablar con Docta | segundos |

Todo menos `ENTRENAR` funciona sin GPU, sin conexión y sin entrenar nada: usa la
base curada. La GPU no acelera nada de momento, porque el forward y el backward
están escritos en numpy y numpy no la usa; para aprovecharla habría que portar
`model.py` a JAX o torch.

### Verificado en Kaggle, no supuesto

El notebook se ejecutó en un kernel de Kaggle real y entrenó:

```
época  40/40  ent=0.1792  val=0.6930  acierto=90.4%   97s
guardado en /kaggle/working/docta/checkpoints/pre_tiny.npz
```

Mejor y más rápido que en local (aquí: val 0.712, acierto 89.8 %, 178 s), que es
justo la razón de irse a Kaggle. Las dos suites pasan dentro del paquete, el
gradiente sale `CORRECTO` con diferencia máxima 7.01e-10, y la puerta responde
las 9 preguntas como debe.

### El editor de Kaggle se bloquea con celdas grandes

La primera versión llevaba los 178 KB en **una sola celda** y Kaggle mostraba
«¡Algo salió mal y esta página se bloqueó!» con este error:

```
NotFoundError: Failed to execute 'removeChild' on 'Node':
The node to be removed is not a child of this node.
    en sp (vendor.js) → sf → s_ → sf → s_ ...
```

Y aquí lo importante: **el notebook no estaba roto**. Se ejecutaba, entrenaba y
daba respuestas correctas. Lo que reventaba era el EDITOR al intentar pintarlo.

La causa no es el tamaño a secas. El blob iba entre comillas triples, así que el
resaltador de sintaxis entraba en modo «cadena que atraviesa 1.740 líneas» y
montaba un árbol de nodos que el DOM virtual no sabía deshacer —la recursión
`sf`/`s_` del stack trace era el resaltador, no el código.

Se arregla con dos cambios que van juntos, y ninguno basta por separado:

- cada línea base64 es una **cadena simple de una línea**, nunca un bloque entre
  comillas triples, para que el resaltador no anide nada;
- el blob se **parte en celdas de ~16 KB** (`TAMANO_TROZO`), 12 celdas, para que
  ninguna sea enorme. La más grande pasa de 178.068 a 18.179 caracteres.

Y se añade una tercera cosa que no tiene nada que ver pero que estaba mal: las
celdas no llevaban `id`, y con `nbformat_minor: 5` eso hoy es un aviso y mañana
será un error duro. Un notebook que el visor del destino rechaza no se ejecuta.

### Plan B: la vía Dataset

Si aun así el editor se queja, `scripts/construir_kaggle.py` deja también
`docta_dataset_kaggle.zip`. Se sube como Dataset privado, se adjunta al notebook
con *Add Input*, y el notebook pasa a tener 8 KB y a ser legible de arriba abajo:

```python
import sys; sys.path.insert(0, "/kaggle/input/docta")
from retriever import Retriever, responder
```

Mismo código, dos formas de llevarlo. Ninguna necesita credenciales dentro del
notebook: el Dataset se sube con la cuenta de cada uno, desde el navegador o con
`kaggle.json` en su propia máquina. Un token pegado en un notebook deja de ser
secreto en cuanto el notebook se comparte, y estos están pensados para
compartirse.

---

## Arquitectura y conocimiento van separados

Cambian a ritmos distintos y por motivos distintos. El código se toca cuando hay
un bug o una mejora del modelo; el conocimiento se toca cada vez que Docta
aprende algo, y eso va a ser constantemente. Tenlos en el mismo paquete y cada
ficha nueva obliga a reconstruir y resubir 200 KB de código que no ha cambiado.

| paquete | qué lleva | cuándo cambia |
|---|---|---|
| `docta-codigo` | módulos, scripts y tests | con un bug o una mejora |
| `docta-conocimiento` | fichas, corpus, sft, tokenizer y manifiesto | cada vez que aprende algo |

El notebook del modo separado pesa **18 KB** y se lee de arriba abajo, frente a
los 227 KB del autocontenido. Sigue existiendo el autocontenido
(`--autocontenido`) para quien no quiera subir datasets.

### Versionar el conocimiento

`scripts/versionar_conocimiento.py` deja una foto en `data/MANIFIESTO.json`: qué
fichas hay, cuántas por tema, una huella de cada archivo y una versión que sube
sola cuando el contenido cambia de verdad.

```
conocimiento v1.1.2 · 53 hechos · huella 04a8dde6f5276c61 · vocab 3008
```

Con `--diff` dice qué cambió desde la última versión. No es control de versiones
—para eso está git—: es un registro que lee el entrenamiento, para saber si hace
falta reentrenar y para anotar en el checkpoint con qué conocimiento se entrenó.

El **tokenizer va con el conocimiento y no con el código**, y es una decisión con
trampa. Es un archivo de datos, pero si se reentrena cambia el vocabulario y con
él la forma de los embeddings: ningún checkpoint anterior vuelve a cargar. El
manifiesto lo vigila aparte y avisa en mayúsculas. Mientras no se toque, el BPE
es a nivel de byte y cualquier texto nuevo en español ya es codificable sin
reentrenar nada, que es justo lo que hace que esto sea limpio.

### Reentrenamiento incremental

```bash
python scripts/entrenar.py --preset tiny --epocas 10 --reanudar checkpoints/pre_tiny
```

Carga el checkpoint y sigue. Dos cosas que no son obvias:

**Se entrena con el corpus COMPLETO, no solo con las fichas nuevas.** Entrenar
solo con lo nuevo es la receta del olvido catastrófico: el modelo ajusta los
pesos para lo que acaba de ver y pierde lo anterior. Aquí el corpus entero cabe
en segundos, así que no hay motivo para arriesgarlo. Cuando no quepa, la solución
es mezclar lo nuevo con una muestra de lo viejo, no entrenar solo con lo nuevo.

**La carga es estricta.** No se reutiliza `Modelo.load`, que hace
`if k in datos: m.p[k] = datos[k]`: se salta en silencio las claves que faltan y
no mira las formas. Para inferir da igual; para reanudar no. Si cambió el
vocabulario, el embedding del archivo tiene otra forma, se asigna tal cual, y el
modelo queda con una matriz que no se corresponde con su propia configuración. No
falla al cargar: falla tres minutos después, en el primer forward, con un error de
dimensiones que no apunta a la causa. Aquí se exige que las claves y las formas
cuadren, y si no, se dice qué no cuadra.

Los momentos de Adam **no** se guardan, así que el optimizador arranca en frío
aunque los pesos vengan calientes. Por eso la primera época de un reentrenamiento
puede empeorar la pérdida antes de mejorarla. No es que esté olvidando.

Cada checkpoint lleva un `.meta.json` con su linaje:

```json
{"conocimiento": "1.1.2", "huella_conocimiento": "04a8dde6f5276c61",
 "hechos": 53, "mejor_val": 3.1134, "reanudado_de": "checkpoints/pre_tiny",
 "conocimiento_anterior": "1.1.2"}
```

Un `.npz` suelto no dice de qué sabe, y esa es exactamente la pregunta que hay
que poder responder cuando el modelo empieza a contestar cosas raras.

### El flujo al añadir conocimiento

```bash
# 1. editar o añadir fichas en data/conocimiento/*.jsonl
python scripts/construir_corpus.py              # 2. regenera corpus y sft
python scripts/versionar_conocimiento.py        # 3. sube versión, deja huella
python scripts/construir_kaggle.py --solo conocimiento   # 4. solo ese paquete
# 5. subir la versión nueva del dataset docta-conocimiento
# 6. en el notebook, la celda REENTRENAR encuentra el checkpoint y continúa
```

Kaggle versiona los datasets por su cuenta, así que queda histórico de qué sabía
el modelo en cada momento.

### Dos bugs que salieron al montar esto

**El corpus no derivaba de la base.** `construir_corpus(base)` recibía la base y
la **ignoraba**: el corpus salía entero de una prosa escrita a mano. Se podía
añadir una ficha, reconstruir todo, reentrenar... y el corpus salía idéntico byte
a byte. El conocimiento nuevo solo entraba por el ajuste fino, nunca por el
pre-entrenamiento. La firma mentía: aceptaba un parámetro que no usaba. Ahora
cada ficha aporta un documento de prosa al corpus.

**Al añadir conocimiento, los tests de "no lo sé" fallan.** Y fallan bien. Son
aserciones negativas atadas al contenido de la base: dicen "esto no lo sabe", y
eso solo es verdad mientras nadie añada la ficha. Al añadir Kazajistán, el test
que exige que no sepa Kazajistán falla. El mensaje ahora distingue los dos casos
que produce ese fallo —"la puerta se ha roto y ahora inventa" (grave) y "has
añadido conocimiento y esta pregunta ya toca contestarla" (esperado)— y dice cuál
de los dos es, en vez de un `¡CONTESTÓ!` que no explica nada.


---

## El temario

`temario.py` recoge qué debe saber Docta, sacado de los dos documentos aportados
al proyecto:

* «Taxonomía Integral de Conocimientos y Competencias Esenciales para el Adulto
  Contemporáneo» — PIAAC y DeSeCo (OCDE), DigComp 2.2, OECD/INFE, las diez
  habilidades de vida de la OMS, UNESCO.
* «Competencias Académicas Fundamentales y Cultura General Sustancial para la
  Ciudadanía en la Sociedad Contemporánea».

Nueve dominios, 31 bloques. Y un informe de cobertura:

```bash
python temario.py            # qué hay curado y qué falta
python temario.py --detalle  # bloque a bloque, con ejemplos de ficha pendiente
```

```
TEMARIO · 9 dominios · 31 bloques · cubiertos 10/31 (32%)

  Matemáticas y razonamiento cuantitativo  ····  0/4      VACÍO
  Ciencias naturales                       █···  1/4    parcial
  Historia y procesos globales             █···  1/4    parcial
  Geografía sistémica y territorial        ███  3/3   completo
  Lengua, literatura y comunicación        █··  1/3    parcial
  Tecnología y competencia digital         █···  1/4    parcial
  Economía y finanzas personales           ····  0/4      VACÍO
  Ética, filosofía y ciudadanía            ██··  2/4    parcial
  Metacognición y límites propios          █  1/1   completo
```

### Por qué los PDF no entran en la base de conocimiento

Se midió antes de decidirlo. Un extractor recorrió los dos documentos buscando
frases que afirmen un dato comprobable —una cifra con su unidad, una fecha, una
relación verificable— y el resultado fue **cero de 279 frases**. El mismo
extractor, sobre las 53 fichas ya curadas, encuentra 41 datos en 110 frases, un
37 %. No es que falle el extractor: es que esos documentos no contienen datos.

Son **temarios, no contenido**. «El ciudadano debe comprender la entropía» no es
una ficha, es un apunte de lo que falta por escribir.

Meterlos en la base sería dañino por dos vías. Como fichas, una frase curricular
recuperada se serviría como respuesta: preguntado por la entropía, Docta
contestaría «el ciudadano debe comprender la irreversibilidad de los procesos
térmicos», que no responde nada y además saldría con una ✓ de verificación,
porque estaría "respaldada" por su propia ficha. Y en el corpus de
pre-entrenamiento enseñarían el registro: Docta acabaría hablando como un informe
de la OCDE, justo lo contrario del carácter sobrio y directo que se decidió.

Por eso los PDF alimentan `temario.py` —la lista de la compra— y no
`data/conocimiento/` —lo que hay en la nevera.

### Un bug de cobertura que salió al hacerlo

El informe casaba las marcas del temario contra el texto de las fichas con
`if marca in texto`, o sea por subcadena. El resultado decía 48 % de cobertura y
era casi todo ruido: "gen" casaba dentro de **gente** y de **general** (13 fichas
de geografía contaban como biología), "iva" dentro de **legislativa** (Canberra y
Pretoria salían como fiscalidad), "bono" dentro de **abono**.

Es literalmente el mismo fallo que ya se había corregido una vez en el
recuperador, donde "iva" vetaba "recursiva". Se volvió a cometer por escribir la
comprobación rápida en vez de buscar la que ya funcionaba. Arreglado con
fronteras de palabra, la cobertura real es 32 %, no 48 %.

Queda un límite que no se puede arreglar casando palabras y que el propio informe
declara: no distingue una ficha que **trata** de un tema de una que lo
**menciona** de pasada. "Cervantes" aparece en una ficha sobre cuánta gente habla
español, y esa ficha no es de literatura. Las marcas con dos sentidos dentro del
propio proyecto ("fuente" significa a la vez procedencia de un dato y medio de
comunicación) se quitaron por inservibles.


---

## La puerta de ignorancia

Cuatro comprobaciones, y basta con que falle **una** para que Docta se niegue a
dar un dato. El orden importa: el motivo que se reporta debe ser el verdadero.

```
0. VETO POR TEMA   ¿la pregunta pide consejo sobre medicina, legal, fiscal,
                   finanzas, programación o actualidad? Se decide sobre la
                   PREGUNTA, antes de mirar la base.
1. EVIDENCIA       ¿el mejor candidato supera el umbral de similitud?
2. FIABILIDAD      ¿la certeza curada del hecho supera el umbral?
3. ENTIDADES       ¿el hecho menciona algún nombre propio de la pregunta?
```

### El veto mira la intención, no solo las palabras

Esto cambió, y el motivo es el temario de competencias del adulto. El veto por
palabras bloqueaba igual «¿qué es el IVA?» —cultura general que cualquier adulto
tiene— que «¿cómo declaro el IVA siendo autónomo?» —un trámite personal donde un
error cuesta dinero. Las dos contienen "iva".

El criterio no está en el tema sino en **a quién se refiere la pregunta**:

| intención | qué es | ejemplo | qué hace Docta |
|---|---|---|---|
| `CONCEPTO` | pregunta por el mundo | «¿qué es el interés compuesto?» | pasa al recuperador |
| `CONSEJO` | pregunta por su propia situación | «¿me conviene refinanciar?» | veta y redirige |
| `ABSOLUTO` | nunca, se formule como se formule | «¿cuál es la dosis letal?» | niega en seco |

Las señales de `CONSEJO` son primera o segunda persona, posesivos y verbos de
decisión aplicados al caso propio (`me conviene`, `debo`, `puedo tomar`,
`mi hijo`, `qué hago`). Las de `CONCEPTO` son los giros de definición: `qué es`,
`cómo funciona`, `qué significa`, `en qué consiste`, `cuál es la diferencia`.

**La ambigüedad se resuelve a `CONSEJO`.** Un veto que ante la duda contesta es
un veto que se salta con cualquier pregunta rara; uno que ante la duda calla
cuesta un "no lo sé" de más, que es exactamente el comportamiento que se busca
aquí.

Dos detalles que no son menores:

**`ABSOLUTO` se comprueba antes que el tema.** Las listas de temas se hicieron
para asesoramientos profesionales y no contienen "bomba" ni "hackear". Si el
tema se mirara primero, «¿cómo puedo fabricar una bomba casera?» saldría por no
casar con ninguno y se contestaría. La pregunta más peligrosa sería justo la que
se cuela.

**Y no lleva redirección.** El veto normal cierra con "pregúntale a quien sepa",
que en medicina o impuestos es lo correcto: hay profesionales cuya función es
saber eso. En lo absoluto esa frase es una instrucción peligrosa —le dice a quien
pregunta que busque a otro que se lo explique. La respuesta es «No. Eso no te lo
voy a explicar.» y nada más, porque cuanto más se extiende una negativa de ese
tipo, más pistas da sobre cómo reformular la pregunta.

Cuando la pregunta es legítima pero Docta no tiene la ficha, el "no lo sé"
**señala a quién preguntar** (`A_QUIEN_PREGUNTAR` en `identidad.py`). Rechazar y
redirigir no son lo mismo, y un sistema que solo sabe rechazar acaba siendo
inútil justo donde más falta hace indicar a quién ir.

Un efecto lateral útil: ahora es seguro **ampliar** las listas de temas. Antes,
añadir "deducción" o "declaración" habría vetado en falso preguntas legítimas.
Con el filtro de intención por delante, ampliar la lista significa acertar más,
no callar más.

La salida no es binaria: es un **grado** que modula el tono.

| grado | condición | cómo habla |
|---|---|---|
| `seguro` | apoyo ≥ 0.75 | afirma sin matices |
| `probable` | apoyo ≥ 0.50 | contesta y marca que el dato es aproximado |
| `dudoso` | apoyo ≥ 0.15 | en condicional, como orientación |
| `no_lo_se` | todo lo demás | lo dice, explica por qué, y señala a dónde mirar |

## La capa de citas

Citar no es poner una bibliografía al pie. Si el sistema cita fuentes mientras
alucina, las citas hacen **más daño** que su ausencia: prestan credibilidad al
invento. Así que la verificación va antes que la cita, y se le aplica también al
texto que genera la red neuronal, no solo al curado.

Cada frase de la respuesta se comprueba contra la evidencia por tres vías, y la
duda siempre resuelve a "sin respaldo":

| señal | qué pilla | ejemplo |
|---|---|---|
| **cifras** | cualquier número que la fuente no contenga, normalizando separadores de miles y decimales | «París tiene 9,2 millones de habitantes» → la fuente no tiene ese número |
| **cobertura léxica** | invención descarada: palabras que no están en ninguna parte | «el volcán Kriskadi» → no casa con nada |
| **pares consecutivos** | relaciones inventadas con vocabulario correcto | véase el fallo 3 más abajo |

La salida lleva marcas por frase —✓ respaldada, ~ en parte, ✗ sin respaldo— y un
bloque de fuentes que dice `comprobada` o `SIN COMPROBAR`. Esa distinción no es
cosmética: `verificado` significa "un humano abrió la fuente y comprobó que el
dato está ahí", no "esto es verdad".

Y si no hay fuentes, se dice:

> Fuentes: sin fuente documentada. Esto viene de la base curada pero nadie ha
> documentado de dónde se sacó.

Hoy los 53 hechos están así. Es el estado honesto de la base y la capa lo
refleja en vez de disimularlo.


---

## Tres fallos de seguridad que aparecieron al probar

### 1. Contestó sobre Brasil cuando le preguntaron por Bután

```
tú    › ¿Qué idioma se habla en Bután?
Docta › Brasil, que habla portugués, y es con diferencia el más grande y
        poblado del continente. Tampoco hablan español Guyana, con inglés...
```

Fluido. Con un dato correcto. Sobre el país equivocado. La similitud léxica era
alta (0,258: comparten "idioma" y "habla") pero **el sujeto de la pregunta no
aparecía en ninguna parte del hecho**.

Es el fallo grave de la bolsa de palabras, y es grave porque *no se parece a un
error*: se parece a una respuesta. Subir el umbral no lo arregla, porque la
similitud era legítimamente alta. Lo arregla exigir que las entidades de la
pregunta aparezcan en el hecho. Ahora dice "no lo sé. No está en lo que he
aprendido."

Y un detalle dentro del detalle: al principio la negativa decía *"tengo algo que
se le parece en las palabras, pero no trata de lo que preguntas"*, que implica
tener algo. Si **ningún** candidato cubre la entidad, la base no tiene nada sobre
eso y hay que decirlo así.

### 2. Se negó a contestar "¿cuál es la capital de Francia?"

El filtro de temas vetados detectaba por **subcadena**: `"iva" in texto`. Y "iva"
está dentro de "recurs**iva**". Resultado: Docta respondía *"de impuestos no
opino"* a preguntas de geografía que sí sabía.

Arreglado detectando por **palabras completas**, con normalización de acentos. Y
se quitaron los nombres de lenguajes de las señales: *"¿cuál es la isla más
grande de Java?"* es geografía legítima y quedaba vetada porque Java es un
lenguaje de programación. Los nombres propios de lenguajes también son islas,
ciudades y personas.

De aquí salió `tests/test_puerta.py`, con **dos listas que pesan lo mismo**: lo
que debe vetar (falsos negativos, peligrosos) y lo que **no** debe vetar (falsos
positivos, que bloquean conocimiento legítimo). Un filtro de seguridad que solo
se mide por lo que atrapa acaba atrapándolo todo.

```
✓ 46 comprobaciones correctas
```

---

### 3. Citó una frase falsa usando solo palabras de la fuente

El test de verificación tenía un caso que debía fallar y fallaba. Sobre la ficha
de Sudáfrica —que dice que tiene tres capitales y que **Ciudad del Cabo** es la
sede del parlamento— se construyó esta respuesta:

> «Sudáfrica tiene tres capitales y Johannesburgo es la sede del parlamento.»

Es falsa. Y el verificador la daba por **respaldada al 100 %**. No le faltaba
razón técnica: todas y cada una de las palabras de la frase aparecen en la
fuente. `Johannesburgo` está, `sede` está, `parlamento` está. Lo único inventado
es **cómo se relacionan**, y contar palabras no puede ver eso.

Este es el fallo más peligroso de los tres, porque los otros dos se notan. Aquí
la IA produce una frase impecable, con el vocabulario exacto de su fuente, y se
equivoca en la única parte que importa.

Se arregló con dos cambios:

**Pares de palabras consecutivas.** La fuente nunca pone `johannesburgo` junto a
`sede`. Una relación inventada produce pares que el texto de referencia no junta
nunca. Con esa señal la frase cae al 0 %.

Y aquí hubo un segundo bug, más tonto y más instructivo: la primera versión
emparejaba **todas** las palabras, incluidas las vacías. La frase falsa pasó de
0 % a 89 % de cobertura, porque los pares `es_la` y `la_sede` aparecen en
cualquier texto del mundo y no discriminan nada. Los pares tienen que ser de
palabras con contenido. Por eso `citacion.pares_de` quita stopwords y
`retriever.bigramas` no: para **verificar** una afirmación las palabras vacías
son ruido, para **puntuar** una consulta afinan el parecido. Dos funciones
parecidas y separadas a propósito.

**Verificar por proposiciones, no por frases.** La frase corrupta entera puntúa
bien porque la mitad correcta arrastra a la falsa. Se corta por comas y
conjunciones, y cada proposición que afirma algo se comprueba por su cuenta;
manda la peor.

Cortar bien es más difícil de lo que parece y costó tres intentos:

- Cortar por "espacio seguido de mayúscula" parte «**El** Everest mide 8.849
  metros» en dos, y verificar cachos de frase no sirve para nada: un cacho
  siempre parece respaldado.
- Proteger las abreviaturas con un `replace("s.", ...)` a pelo se carga las
  frases enteras, porque `s.` es subcadena de **"metros."**. Protegía el punto
  que cerraba la frase y el separador dejaba de cortar. Las abreviaturas hay que
  anclarlas.
- Cortar por comas sin más parte los decimales: «44,6 millones» se convierte en
  «Asia tiene unos 44» y «6 millones de kilómetros cuadrados», y entonces el
  verificador busca la cifra 44, que no está en ninguna fuente. **El corte
  fabricaba un falso positivo sobre un dato correcto.**

Y una "y" no separa siempre dos proposiciones. «entre Nepal **y** el Tíbet» es
una enumeración; «tres capitales **y** Johannesburgo **es** la sede» son dos
afirmaciones. El criterio es si hay verbo conjugado a los dos lados —la única
pieza de análisis sintáctico de todo el proyecto, y son 84 formas verbales en una
lista.

Lo que la capa sigue sin poder detectar: una relación falsa expresada sin
números y con palabras que la fuente sí junta en otro contexto. Eso requiere
entender el texto, no compararlo. Está acotado en los tests como limitación
conocida, no como comportamiento correcto.


## Tres bugs del backward que aparecieron y lo que enseñan

Los tres estaban en el backward y los tres **entrenaban sin dar error**. Solo la
comprobación numérica los destapó.

**1. Confundir el gradiente de la entrada normalizada con el de la salida del
bloque.** La RMSNorm no está en el camino residual: es una rama que sale de
`x_mid` y vuelve a sumarse. Bajando desde `dx`, lo primero es que
`dL/d(ffn_out) = dx` directamente, y solo después se atraviesa la norma. Al
revés, el FFN recibía un gradiente ~400× mayor (el factor es del orden de
`1/rms²`). La pérdida bajaba igual.

**2. `dwo = ctx.T @ do`, no `do.T @ ctx`.** Para `y = x @ W` el gradiente es
`x.T @ dy`. Pero `wo` es **siempre cuadrada** (`d_model == n_heads × head_dim`
por construcción), así que escribirla al revés no da error de formas: da la
matriz transpuesta. Las demás proyecciones no son cuadradas y ahí el mismo error
sí revienta. Ahora hay una aserción que compara la forma de cada gradiente con
la de su peso.

**3. Probar con `ffn_hidden == d_model`.** En mis tests de aislamiento puse las
dimensiones iguales y las matrices cuadradas hacían que todo encajara por pura
casualidad. Regla: en un test de gradientes, **todas las dimensiones deben ser
distintas entre sí**. Ahora `d_model=32`, `ffn_hidden=20`, `moe_ffn_hidden=12`,
`kv_dim=16`.

Y uno que no era bug pero lo parecía: al comprobar gradientes en float32 con
`h=1e-4` el numérico salía 0. La pérdida está cuantizada (el salto mínimo entre
dos float32 cercanos a 6 es ~5e-7), así que la perturbación no llegaba a cambiar
el valor. La comprobación se hace en **float64**; el entrenamiento sigue en
float32, que es la mitad de memoria.

---

## Siguiente paso

1. **Ampliar el corpus.** Es el cuello de botella real. Con 2.264 tokens no hay
   nada que aprender. Hace falta más prosa y más hechos curados, y subir la
   proporción de ejemplos "no lo sé" en el SFT, que ahora es del 6 % — demasiado
   baja para que la conducta de admitir ignorancia se consolide frente a la de
   contestar siempre.
2. **Generación y chat.** Muestreo con temperatura, bucle conversacional, y que
   el grado de la puerta (`seguro` / `probable` / `dudoso` / `no_lo_se`) module
   el tono de la respuesta.
3. **Memoria episódica** y aprendizaje de las correcciones.
4. **Bucle de sueños** con el árbol de descubrimientos.

## Lo que falta de la arquitectura de DeepSeek, a propósito

* **Multi-head Latent Attention (MLA)**: comprime la cache de clave/valor en un
  vector latente. Es una optimización de memoria para inferencia larga; con
  ctx=256 no cambia nada observable y complica el backward muchísimo.
* **Multi-Token Prediction**: predecir varios tokens por paso. Útil a escala,
  ruido a esta escala.

Ambos son añadidos locales cuando el modelo sea más grande y el contexto largo.
