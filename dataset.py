"""
Construcción de lotes de entrenamiento.

Dos regímenes, porque son dos objetivos distintos:

  Pre-entrenamiento (sobre corpus.txt)
    Ventanas de tokens consecutivos. Se predice cada token a partir de los
    anteriores y TODAS las posiciones cuentan para la pérdida. De aquí sale el
    idioma: cómo se construye una frase en español.

  Ajuste fino (sobre sft.jsonl)
    Pares pregunta/respuesta en formato de chat. Y aquí viene lo importante:
    la pérdida se calcula SOLO sobre los tokens de la respuesta. El modelo no
    debe aprender a generar la pregunta del usuario, debe aprender a generar la
    respuesta. Si se entrena con todo, desperdicia capacidad imitando al usuario
    y además la pérdida queda dominada por texto que no le interesa.

    Y en los ejemplos "no_sabe" la respuesta es una negativa. Eso es justo lo
    que queremos que aprenda a producir cuando no tiene contexto: la conducta
    de admitir ignorancia se entrena, no se improvisa.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from tokenizer import BPETokenizer, BOT, CTX_OPEN, CTX_CLOSE, EOS, PAD, USER

AQUI = Path(__file__).resolve().parent


class LotesPreentrenamiento:
    """Ventanas deslizantes sobre el corpus, barajadas en cada época."""

    def __init__(self, tokenizador: BPETokenizer, ruta_corpus: Path,
                 ctx: int = 256, solape: int = 32, semilla: int = 0):
        texto = Path(ruta_corpus).read_text(encoding="utf-8")
        self.ids = np.array(tokenizador.encode(texto, add_specials=False), dtype=np.int64)
        self.ctx = ctx
        self.paso = max(1, ctx - solape)
        self.rng = np.random.default_rng(semilla)
        self.n_ventanas = max(0, (len(self.ids) - 1 - ctx) // self.paso + 1)
        if self.n_ventanas == 0:
            raise ValueError(
                f"El corpus da {len(self.ids)} tokens, insuficiente para "
                f"ventanas de {ctx}. Hace falta más texto o menos contexto.")

    def __len__(self) -> int:
        return self.n_ventanas

    def epoca(self, batch_size: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Contrato de formas. Conviene tenerlo por escrito porque es la causa
        número uno de errores tontos en estos bucles:

            se rinde  entrada (B, L)   y   mascara (B, L-1)

            mascara[b, t] = 1 significa "la predicción de la posición t cuenta
            para la pérdida", y esa predicción es el token entrada[b, t+1].
            El modelo recibe ids (B,T) con mask (B,T-1), hace targ = ids[:,1:]
            y lo compara contra logits[:,:-1]. Así las longitudes encajan.
        """
        orden = self.rng.permutation(self.n_ventanas)
        for ini in range(0, len(orden), batch_size):
            idx = orden[ini:ini + batch_size]
            L = self.ctx
            entrada = np.full((len(idx), L), PAD, dtype=np.int64)
            mascara = np.zeros((len(idx), L - 1), dtype=np.float32)
            for i, s in enumerate(idx * self.paso):
                trozo = self.ids[s:s + L]
                entrada[i, :len(trozo)] = trozo
                # en pre-entrenamiento cuenta toda posición con token siguiente
                mascara[i, :max(0, len(trozo) - 1)] = 1.0
            if mascara.sum() == 0:
                continue
            yield entrada, mascara


class LotesSFT:
    """
    Pares pregunta/respuesta. La máscara vale 1 SOLO en los tokens de la
    respuesta: es lo único cuya generación queremos aprender.
    """

    def __init__(self, tokenizador: BPETokenizer, ruta_sft: Path,
                 ctx: int = 256, semilla: int = 0):
        self.tok = tokenizador
        self.ctx = ctx
        self.rng = np.random.default_rng(semilla)
        self.descartados = 0        # ejemplos que no caben en ctx
        self.ejemplos: list[dict] = []
        self.ejemplos_codificados: list[tuple[list[int], list[int]]] = []

        for linea in Path(ruta_sft).read_text(encoding="utf-8").splitlines():
            linea = linea.strip()
            if not linea:
                continue
            e = json.loads(linea)
            ids, mascara = self._codificar(e["texto"])
            if ids is None:
                continue
            self.ejemplos.append(e)
            self.ejemplos_codificados.append((ids, mascara))

    def _codificar(self, texto: str) -> tuple[Optional[list[int]], Optional[list[int]]]:
        """
        Codifica marcando dónde empieza la respuesta.

        La respuesta es lo que va DESPUÉS del último <bot>. Se busca el token
        por su id, no por texto, para que no haya ambigüedad si el contenido
        contiene algo parecido.

        Los ejemplos que no caben en ctx se DESCARTAN enteros, no se recortan:
        una respuesta truncada es una respuesta sin <eos>, y eso enseña justo
        lo que no queremos, que el modelo no sepa cuándo parar de hablar.
        """
        ids = self.tok.encode(texto, add_specials=False)
        if BOT not in ids:
            return None, None
        if len(ids) > self.ctx:
            self.descartados += 1
            return None, None

        corte = len(ids) - 1 - ids[::-1].index(BOT)     # último <bot>
        if corte + 1 >= len(ids):
            return None, None
        # mascara[t] = 1 si la posición t (que predice ids[t+1]) está en la respuesta
        mascara = [0.0] * len(ids)
        for t in range(corte, len(ids) - 1):
            mascara[t] = 1.0
        if sum(mascara) == 0:
            return None, None
        return ids, mascara

    def __len__(self) -> int:
        return len(self.ejemplos_codificados)

    @property
    def proporcion_no_sabe(self) -> float:
        n = sum(1 for e in self.ejemplos if e.get("tipo") == "no_sabe")
        return n / max(len(self.ejemplos), 1)

    def epoca(self, batch_size: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Mismo contrato de formas que preentrenamiento: se rinde entrada (B, L)
        y mascara (B, L-1). La máscara de la última posición no existe porque
        no hay token siguiente que predecir. El relleno no cuenta para la
        pérdida (máscara 0), que si no el modelo aprendería a decir <pad>.
        """
        orden = self.rng.permutation(len(self.ejemplos_codificados))
        for ini in range(0, len(orden), batch_size):
            lote = [self.ejemplos_codificados[i] for i in orden[ini:ini + batch_size]]
            ancho = max(len(ids) for ids, _ in lote)
            ids_b = np.full((len(lote), ancho), PAD, dtype=np.int64)
            mas_b = np.zeros((len(lote), ancho), dtype=np.float32)
            for i, (ids, mascara) in enumerate(lote):
                ids_b[i, :len(ids)] = ids
                mas_b[i, :len(mascara)] = mascara
            yield ids_b, mas_b[:, :-1]


if __name__ == "__main__":
    tok = BPETokenizer.load(AQUI / "data" / "tokenizer.json")
    print(f"tokenizer: vocabulario de {len(tok)}\n")

    pre = LotesPreentrenamiento(tok, AQUI / "data" / "corpus.txt", ctx=128)
    print(f"pre-entrenamiento: {len(pre.ids):,} tokens -> {len(pre)} ventanas de 128")
    for ids, mask in pre.epoca(4):
        print(f"  lote {ids.shape} máscara {mask.shape} posiciones válidas "
              f"{int(mask.sum())}/{mask.size}")
        print(f"  primer ejemplo decodificado: {tok.decode(ids[0][:60])!r}...")
        break

    sft = LotesSFT(tok, AQUI / "data" / "sft.jsonl", ctx=256)
    print(f"\najuste fino: {len(sft)} ejemplos "
          f"({100*sft.proporcion_no_sabe:.0f} % son respuestas de 'no lo sé')")
    for ids, mask in sft.epoca(4):
        print(f"  lote {ids.shape} | tokens de respuesta {int(mask.sum())} "
              f"de {mask.size} ({100*mask.mean():.0f} %)")
        i = 0
        completo = ids[i].tolist()
        print(f"  -- entrada : {tok.decode(completo)[:110]!r}")
        solo_resp = [t for t, mv in zip(completo, mask[i]) if mv > 0]
        print(f"  -- respuesta supervisada: {tok.decode(solo_resp)[:110]!r}")
        break
