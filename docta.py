#!/usr/bin/env python3
"""
Docta — punto de entrada.

Uso:
    python docta.py                    conversación interactiva
    python docta.py -p "tu pregunta"   una sola pregunta
    python docta.py --quien            quién es y qué sabe
    python docta.py --tripa            enseña las puntuaciones internas
    python docta.py --modelo checkpoints/sft_tiny.npz
                                       habla la red neuronal (RAG + cinturón)
    python docta.py --citas            marca cada frase con su respaldo

Comandos dentro de la conversación:
    /quien     presentación
    /temas     qué materias tiene curadas
    /limite    hasta cuándo sabe cosas
    /salir

Por defecto responde DIRECTAMENTE desde la base curada, sin pasar por la red
neuronal: con un modelo de 7M parámetros, dejarle reescribir un dato correcto
solo puede empeorarlo. Con --modelo la red sí habla, pero no suelta: el
contexto se le inyecta recuperado de la base (nunca lo genera ella) y todo lo
que produzca se verifica contra la evidencia antes de enseñarlo. Si la
verificación no pasa, se sirve la ficha curada tal cual. La decisión de
SABER o NO SABER sigue siendo de la puerta, nunca de la red.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from identidad import NOMBRE, construir_identidad            # noqa: E402
from knowledge import BaseConocimiento                       # noqa: E402
from retriever import Retriever, formular_ignorancia          # noqa: E402

MARCA = {
    "seguro": "  ",
    "probable": "~ ",
    "dudoso": "? ",
    "no_lo_se": "· ",
}


class Docta:
    """Fachada: junta identidad, base de conocimiento y puerta de ignorancia."""

    def __init__(self, umbral_apoyo: float = 0.15, modelo: str | None = None,
                 temperatura: float = 0.7, citas: bool = False,
                 semilla: int = 1234):
        self.base = BaseConocimiento.cargar()
        self.idn = construir_identidad(self.base)
        self.busca = Retriever(self.base, umbral_apoyo=umbral_apoyo)
        self.turnos = 0
        self.temperatura = temperatura
        self.citas = citas
        self.semilla = semilla
        self.modelo = None
        self.tok = None
        self.info_modelo = ""
        self._nota_neural = ""      # qué hizo la red en el último turno
        if modelo:
            self._cargar_modelo(modelo)

    # ------------------------------------------------------------------ #
    def _cargar_modelo(self, ruta: str) -> None:
        """
        Carga red + tokenizer para el modo neural.

        Los imports van aquí dentro y no arriba: sin --modelo, docta.py no
        necesita numpy para nada, y mantener la versión de solo-recuperación
        arrancando en cualquier sitio sin dependencias es una decisión, no un
        descuido.
        """
        import json as _json

        from model import Modelo
        from tokenizer import BPETokenizer

        ruta = Path(ruta).with_suffix("")        # acepta con o sin .npz
        if not ruta.with_suffix(".npz").exists():
            raise SystemExit(f"no encuentro el checkpoint {ruta}.npz")
        self.tok = BPETokenizer.load(Path(__file__).parent / "data" / "tokenizer.json")
        self.modelo = Modelo.load(ruta)
        if self.modelo.cfg.vocab_size != len(self.tok):
            raise SystemExit(
                f"el modelo habla otro idioma: vocabulario del checkpoint "
                f"{self.modelo.cfg.vocab_size} != tokenizer {len(self.tok)}. "
                f"Entrena con el tokenizer actual o usa el suyo.")
        self.info_modelo = str(ruta.name)
        meta = ruta.with_suffix(".meta.json")
        if meta.exists():
            m = _json.loads(meta.read_text(encoding="utf-8"))
            self.info_modelo += (f"  (val {m.get('mejor_val', float('nan')):.3f}"
                                 f" · conocimiento v{m.get('conocimiento', '?')}"
                                 f" · {m.get('hechos', '?')} hechos)")

    # ------------------------------------------------------------------ #
    def responder(self, pregunta: str, tripa: bool = False) -> str:
        """
        El camino completo: recuperar -> puerta de ignorancia -> tono.

        Devuelve texto plano. La decisión de SABER o NO SABER se toma antes de
        generar una sola palabra, que es el orden correcto: si se decide
        después, el sistema ya se ha enamorado de su propia respuesta.
        """
        pregunta = pregunta.strip()
        if not pregunta:
            return ""

        r = self.busca.consultar(pregunta)
        self.turnos += 1
        self._nota_neural = ""

        if not r.sabe:
            # La puerta dice NO: aquí la red no opina. El silencio es una
            # decisión del sistema de evidencia, no un fallo de generación, y
            # delegarlo en el modelo sería justo el error que Docta existe
            # para no cometer.
            salida = formular_ignorancia(r)
        elif self.modelo is not None:
            salida = self._responder_neural(r)
        else:
            salida = self.idn.respuesta_sin_modelo(r.grado, r.mejor.hecho.respuesta)
            if self.citas and r.verificado is not None:
                from citacion import renderizar
                salida = renderizar(r.verificado)

        if tripa:
            salida += self._diagnostico(r)
        return salida

    # ------------------------------------------------------------------ #
    def _responder_neural(self, r) -> str:
        """
        La red habla, pero con cinturón.

        El prompt tiene EXACTAMENTE el formato con el que se entrenó el SFT:

            <user>{pregunta}<eos><bot><ctx>{respuesta curada}</ctx>

        El contexto se le inyecta; nunca lo genera ella. Un modelo que se
        inventa su propio contexto es un modelo que alucina dos veces: una al
        escribir la "evidencia" y otra al citarla.

        Lo generado se verifica contra las fichas recuperadas con el mismo
        verificador que se usa para las respuestas curadas. Si la red tuerce
        un número o añade algo que la evidencia no sostiene, su texto se tira
        a la basura y se sirve la ficha tal cual. Con 7M parámetros la red no
        aporta información: su trabajo es aprender a hablar con el material
        que se le da, y el cinturón garantiza que cuando no lo consiga, no
        pase nada.
        """
        from citacion import Respaldo, Verificador, renderizar
        from tokenizer import (BOT, CTX_CLOSE, CTX_OPEN, DOC, EOS, IDK, PAD,
                               USER)

        ctx = r.mejor.hecho.respuesta
        curada = self.idn.respuesta_sin_modelo(r.grado, ctx)

        prompt = f"<user>{r.consulta}<eos><bot><ctx>{ctx}</ctx>"
        ids = self.tok.encode(prompt, add_specials=False)
        if len(ids) >= self.modelo.cfg.max_seq_len:
            self._nota_neural = "prompt más largo que el contexto; sirvo la ficha"
            return curada

        # De los tokens especiales solo <eos> puede generarse: la red debe
        # saber cuándo parar, pero no soltar andamio de formato.
        nuevos = self.modelo.generar(
            ids, eos=EOS, max_nuevos=200, temperatura=self.temperatura,
            top_k=8, semilla=self.semilla + self.turnos,
            suprimir=(PAD, DOC, USER, BOT, CTX_OPEN, CTX_CLOSE, IDK))
        texto = self.tok.decode(nuevos, skip_specials=True).strip()

        if not texto:
            self._nota_neural = "la red no llegó a decir nada; sirvo la ficha"
            return curada

        ver = Verificador().verificar(texto, [c.hecho for c in r.candidatos])
        sin_respaldo = [a for a in ver.afirmaciones
                        if a.respaldo is Respaldo.NO or a.numeros_faltan]
        if sin_respaldo:
            # Nota honesta en --tripa: la red dijo algo y se descartó. Esto no
            # se oculta porque es justo la métrica que importa: cuántas veces
            # el cinturón salva de la red.
            self._nota_neural = (
                "la red dijo algo sin respaldo y se descartó; sirvo la ficha\n"
                f"  └ descartado: {texto[:120]!r}")
            return curada

        self._nota_neural = f"la red generó la respuesta ({len(nuevos)} tokens)"
        if self.citas:
            return renderizar(ver)
        return texto

    def _diagnostico(self, r) -> str:
        """Lo que hay debajo: útil para ajustar umbrales sin adivinar."""
        lineas = [
            "",
            f"  ┌ grado={r.grado}  apoyo={r.apoyo:.3f}  "
            f"rechazo={r.motivo_rechazo or '—'}",
        ]
        if self._nota_neural:
            nota = self._nota_neural.split("\n")
            lineas.append(f"  │ red: {nota[0]}")
            lineas.extend(f"  {l}" for l in nota[1:])
        for c in r.candidatos[:3]:
            lineas.append(
                f"  │ {c.hecho.id:16} sim={c.similitud:.3f} "
                f"certeza={c.hecho.certeza:.2f} {' '.join(c.motivos)}")
        if not r.candidatos:
            lineas.append("  │ (ningún candidato por encima de cero)")
        lineas.append("  └")
        return "\n".join(lineas)

    # ------------------------------------------------------------------ #
    def quien(self) -> str:
        return self.idn.presentacion()

    def temas(self) -> str:
        stats = self.base.estadisticas()
        lineas = [f"{self.idn.total_hechos} hechos curados, en:"]
        por_tema: dict[str, int] = {}
        for h in self.base:
            if h.nivel != "meta":
                por_tema[h.tema] = por_tema.get(h.tema, 0) + 1
        for tema in sorted(por_tema):
            lineas.append(f"  {tema:34} {por_tema[tema]:>3}")
        lineas.append(f"\nCerteza media de los datos: {stats['certeza_media']}")
        lineas.append(f"Lo más flojo que tengo: {stats['certeza_min']}")
        return "\n".join(lineas)

    def limite(self) -> str:
        lim = self.base.por_tema("meta/limites")
        if not lim:
            return f"Conocimiento curado hasta el {self.idn.corte}."
        return (f"Conocimiento curado hasta el {self.idn.corte}.\n\n"
                + lim[0].respuesta)


# ---------------------------------------------------------------------------
def conversacion(d: Docta, tripa: bool = False) -> None:
    print(d.quien())
    print("\nEscribe una pregunta. /quien /temas /limite /salir\n")
    print("─" * 66)

    while True:
        try:
            pregunta = input("\ntú › ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not pregunta:
            continue

        cmd = pregunta.lower()
        if cmd in ("/salir", "/quit", "salir", "adiós", "adios"):
            break
        if cmd in ("/quien", "/quién"):
            print(f"\n{NOMBRE} › {d.quien()}")
            continue
        if cmd == "/temas":
            print(f"\n{NOMBRE} › {d.temas()}")
            continue
        if cmd == "/limite":
            print(f"\n{NOMBRE} › {d.limite()}")
            continue

        print(f"\n{NOMBRE} › {d.responder(pregunta, tripa=tripa)}")

    print(f"\n{d.turnos} preguntas. Se acabó.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=f"{NOMBRE}: {__doc__}",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-p", "--pregunta", action="append", default=None,
                    help="pregunta suelta (se puede repetir)")
    ap.add_argument("--quien", action="store_true", help="quién es y qué sabe")
    ap.add_argument("--temas", action="store_true", help="materias curadas")
    ap.add_argument("--limite", action="store_true", help="fecha de corte")
    ap.add_argument("--tripa", action="store_true",
                    help="enseña puntuaciones y candidatos internos")
    ap.add_argument("--umbral", type=float, default=0.15,
                    help="apoyo mínimo para atreverse a contestar")
    ap.add_argument("--modelo", default=None, metavar="RUTA",
                    help="checkpoint (.npz) para que hable la red neuronal "
                         "en vez de servir la ficha curada tal cual")
    ap.add_argument("--temperatura", type=float, default=0.7,
                    help="0 = voraz y repetible; ~0.7 = conversación; "
                         ">1 = la red empieza a inventar")
    ap.add_argument("--citas", action="store_true",
                    help="marca cada frase con su respaldo en la evidencia")
    a = ap.parse_args()

    d = Docta(umbral_apoyo=a.umbral, modelo=a.modelo,
              temperatura=a.temperatura, citas=a.citas)
    if d.modelo is not None:
        print(f"[red cargada: {d.info_modelo} · T={a.temperatura}]",
              file=sys.stderr)

    if a.quien:
        print(d.quien()); return
    if a.temas:
        print(d.temas()); return
    if a.limite:
        print(d.limite()); return

    if a.pregunta:
        for q in a.pregunta:
            print(f"tú › {q}")
            print(f"{NOMBRE} › {d.responder(q, tripa=a.tripa)}")
            print()
        return

    conversacion(d, tripa=a.tripa)


if __name__ == "__main__":
    main()
