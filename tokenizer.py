"""
Tokenizer BPE (Byte-Pair Encoding) escrito desde cero.

Por qué a nivel de BYTES y no de caracteres:
  Porque así el tokenizer NUNCA se encuentra un símbolo desconocido. Cualquier
  texto del mundo es una secuencia de bytes UTF-8, y nosotros partimos de los
  256 bytes posibles. Acentos, ñ, emoji, chino: todo entra. Un tokenizer de
  caracteres tendría que reservar un token <unk> y perder información; este no.

El coste es que con un vocabulario pequeño las palabras largas se parten en
muchos trozos. Es la compensación real de los modelos chicos y se ve clarísimo
en `estadisticas()`.

Cómo funciona BPE:
  1. Empezamos con el texto partido en bytes sueltos.
  2. Contamos qué pares de tokens adyacentes aparecen más.
  3. Fusionamos el par más frecuente en un token nuevo.
  4. Repetimos hasta llegar al tamaño de vocabulario deseado.
  El resultado: lo que aparece mucho en el corpus (" que", "ción", "ar") se
  vuelve un solo token; lo raro se queda partido en bytes.
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

# ---------------------------------------------------------------------------
# Tokens especiales. Reservamos los primeros IDs y NUNCA se generan por BPE.
# ---------------------------------------------------------------------------
SPECIAL_TOKENS: list[str] = [
    "<pad>",   # relleno para igualar longitudes en un lote
    "<eos>",   # fin de secuencia
    "<doc>",   # separador entre documentos del corpus de preentrenamiento
    "<user>",  # turno del usuario
    "<bot>",   # turno de la IA
    "<ctx>",   # abre el contexto recuperado de la base de conocimiento
    "</ctx>",  # cierra el contexto recuperado
    "<nosé>",  # la IA admite ignorancia (sí, con tilde: es nuestro token estrella)
]

PAD, EOS, DOC, USER, BOT, CTX_OPEN, CTX_CLOSE, IDK = range(len(SPECIAL_TOKENS))
N_SPECIAL = len(SPECIAL_TOKENS)

# Pre-tokenización estilo GPT-2: partimos por palabras/números/puntuación de
# forma que las fusiones no crucen fronteras raras. Añadimos ñ y tildes al
# grupo de "letras" para que el español no se parta mal.
_SPLIT_RE = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d"""      # contracciones inglesas
    r"""|[\w\d]+"""                      # palabras y números (incluye ñ, á, é...)
    r"""|[^\s\w\d]+"""                   # bloques de puntuación
    r"""|\s+""",                         # espacios (importante: van PEGADOS a la palabra siguiente)
    re.UNICODE,
)


def _pretokenize(text: str) -> list[str]:
    """Trocea el texto en unidades dentro de las cuales sí aplicaremos BPE."""
    return _SPLIT_RE.findall(text)


class BPETokenizer:
    def __init__(self) -> None:
        self.merges: list[tuple[int, int]] = []   # reglas de fusión, en orden
        self.vocab: dict[int, bytes] = {}         # id -> bytes
        self.token2id: dict[bytes, int] = {}      # bytes -> id
        self.special2id: dict[str, int] = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
        self._trained = False

    # ------------------------------------------------------------------ #
    # Entrenamiento                                                       #
    # ------------------------------------------------------------------ #
    def train(self, texts: Iterable[str], vocab_size: int = 4096,
              verbose: bool = True) -> "BPETokenizer":
        """Aprende las fusiones BPE a partir de un corpus de textos."""
        self.vocab = {i: t.encode("utf-8") for i, t in enumerate(SPECIAL_TOKENS)}
        for b in range(256):                                   # los 256 bytes base
            self.vocab[N_SPECIAL + b] = bytes([b])
        self.token2id = {v: k for k, v in self.vocab.items()}
        self.merges = []

        # Contamos los pares una sola vez por pretoken y reutilizamos:
        # mucho más rápido que recontar todo el corpus en cada iteración.
        pair_counts: Counter[tuple[int, int]] = Counter()
        words: Counter[tuple[int, ...]] = Counter()

        for text in texts:
            for chunk in _pretokenize(text):
                ids = tuple(self.token2id[bytes([b])] for b in chunk.encode("utf-8"))
                if len(ids) < 2:
                    continue
                words[ids] += 1
                for a, b_ in zip(ids, ids[1:]):
                    pair_counts[(a, b_)] += 1

        n_merges = vocab_size - len(self.vocab)
        if n_merges <= 0:
            raise ValueError(
                f"vocab_size={vocab_size} demasiado pequeño: los 256 bytes + "
                f"{N_SPECIAL} especiales ya ocupan {len(self.vocab)}"
            )

        for step in range(n_merges):
            if not pair_counts:
                if verbose:
                    print(f"[bpe] sin más pares que fusionar en el paso {step}")
                break
            # Desempate determinista: a igual frecuencia, el par con IDs menores.
            best = min(pair_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            new_id = len(self.vocab)
            self.merges.append(best)
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            self.token2id[self.vocab[new_id]] = new_id

            # Reescribimos las palabras afectadas y actualizamos los conteos.
            new_pair_counts: Counter[tuple[int, int]] = Counter()
            new_words: Counter[tuple[int, ...]] = Counter()
            for ids, freq in words.items():
                if best not in zip(ids, ids[1:]):
                    new_words[ids] += freq
                    for a, b_ in zip(ids, ids[1:]):
                        new_pair_counts[(a, b_)] += freq
                    continue
                merged: list[int] = []
                i = 0
                while i < len(ids):
                    if i < len(ids) - 1 and (ids[i], ids[i + 1]) == best:
                        merged.append(new_id)
                        i += 2
                    else:
                        merged.append(ids[i])
                        i += 1
                t = tuple(merged)
                new_words[t] += freq
                for a, b_ in zip(t, t[1:]):
                    new_pair_counts[(a, b_)] += freq
            words, pair_counts = new_words, new_pair_counts

            if verbose and (step + 1) % max(1, n_merges // 10) == 0:
                tok = self.vocab[new_id].decode("utf-8", errors="replace")
                print(f"[bpe] {step+1:>5}/{n_merges} fusiones "
                      f"| vocab={len(self.vocab)} | último: {tok!r}")

        self._trained = True

        # El corpus puede no dar para llegar al vocabulario pedido: cuando se
        # acaban los pares repetidos, se acabaron las fusiones. No es un fallo,
        # es información: significa que hemos pedido más vocabulario del que
        # este texto puede justificar. Lo avisamos en vez de callarlo.
        if verbose:
            if len(self.vocab) < vocab_size:
                print(f"[bpe] AVISO: el corpus solo da para {len(self.vocab)} tokens "
                      f"de los {vocab_size} pedidos. Con más texto se llenaría.")
            else:
                print(f"[bpe] vocabulario completo: {len(self.vocab)} tokens")

        # Redondeamos hacia arriba a múltiplo de 64. Las matrices de embedding y
        # de salida quedan alineadas y las multiplicaciones van más rápidas; los
        # IDs sobrantes simplemente no se usan nunca.
        self.rellenar_a_multiplo(64)
        return self

    def rellenar_a_multiplo(self, n: int = 64) -> None:
        """Reserva IDs vacíos hasta que el vocabulario sea múltiplo de n."""
        objetivo = ((len(self.vocab) + n - 1) // n) * n
        for tid in range(len(self.vocab), objetivo):
            self.vocab[tid] = b""          # slot reservado, nunca se emite
        return None

    def vocab_size_real(self) -> int:
        """Tokens utilizables (sin contar los huecos de relleno)."""
        return sum(1 for v in self.vocab.values() if v)

    # ------------------------------------------------------------------ #
    # Codificación / decodificación                                       #
    # ------------------------------------------------------------------ #
    def _apply_merges(self, ids: list[int]) -> list[int]:
        """Aplica las reglas de fusión aprendidas, en el orden en que se aprendieron."""
        rank = {pair: i for i, pair in enumerate(self.merges)}
        ids = list(ids)
        while len(ids) >= 2:
            # Buscamos la fusión con MENOR rango = la que aprendimos antes.
            # Ese orden importa: reproduce exactamente el árbol del entrenamiento.
            pairs = [(rank.get((ids[i], ids[i + 1]), 1 << 30), i) for i in range(len(ids) - 1)]
            best_rank, best_i = min(pairs)
            if best_rank == 1 << 30:
                break                                   # ya no hay nada que fusionar
            ids[best_i:best_i + 2] = [self.token2id[self.vocab[ids[best_i]] + self.vocab[ids[best_i + 1]]]]
        return ids

    def encode(self, text: str, add_specials: bool = True) -> list[int]:
        """Texto -> IDs. Los tokens especiales escritos literalmente se respetan."""
        if not text:
            return []
        # Partimos por los tokens especiales para no destruirlos con el BPE.
        pattern = "(" + "|".join(re.escape(t) for t in SPECIAL_TOKENS) + ")"
        out: list[int] = []
        for piece in re.split(pattern, text):
            if not piece:
                continue
            if piece in self.special2id:
                out.append(self.special2id[piece])
                continue
            for chunk in _pretokenize(piece):
                ids = [self.token2id[bytes([b])] for b in chunk.encode("utf-8")]
                out.extend(self._apply_merges(ids))
        if add_specials:
            out.append(EOS)
        return out

    def encode_batch(self, texts: Iterable[str], max_len: Optional[int] = None,
                     pad: bool = True) -> tuple[list[list[int]], list[list[int]]]:
        """
        Codifica varios textos y los deja en matriz rectangular.
        Devuelve (ids, máscara). máscara=1 donde hay token real, 0 donde hay relleno.
        La máscara es imprescindible: sin ella el relleno contamina la pérdida.
        """
        seqs = [self.encode(t) for t in texts]
        if max_len is not None:
            seqs = [s[:max_len] for s in seqs]
        width = max((len(s) for s in seqs), default=0)
        if not pad:
            return seqs, [[1] * len(s) for s in seqs]
        ids = [s + [PAD] * (width - len(s)) for s in seqs]
        mask = [[1] * len(s) + [0] * (width - len(s)) for s in seqs]
        return ids, mask

    def decode(self, ids: Iterable[int], skip_specials: bool = True) -> str:
        """IDs -> texto. Se toleran bytes inválidos a medias (típico al generar)."""
        buf = bytearray()
        for i in ids:
            if i < N_SPECIAL:
                if not skip_specials:
                    buf += SPECIAL_TOKENS[i].encode("utf-8")
                continue
            if i in self.vocab:
                buf += self.vocab[i]
        return buf.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ #
    # Utilidades                                                          #
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return PAD

    @property
    def eos_id(self) -> int:
        return EOS

    def estadisticas(self, muestra: Iterable[str]) -> dict:
        """Diagnóstico honesto de cuánto comprime el tokenizer."""
        chars = toks = 0
        for t in muestra:
            chars += len(t.encode("utf-8"))
            toks += len(self.encode(t, add_specials=False))
        return {
            "vocab_size": len(self.vocab),
            "fusiones": len(self.merges),
            "bytes_por_token": round(chars / max(toks, 1), 3),
            "chars_por_token": round(chars / max(toks, 1), 3),
            "bytes_analizados": chars,
            "tokens_generados": toks,
        }

    # ------------------------------------------------------------------ #
    # Persistencia                                                        #
    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        """
        Guarda el vocabulario.

        Detalle que rompe si no se piensa: los tokens son BYTEs arbitrarios, y
        muchos no forman UTF-8 válido (b'\\x80' suelto, por ejemplo). No se
        pueden serializar como texto. Los guardamos con latin-1 + surrogateescape,
        que mapea cada byte a un codepoint y es perfectamente reversible.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", errors="surrogateescape") as f:
            json.dump({
                "version": 1,
                "merges": [list(m) for m in self.merges],
                "vocab": {str(k): v.decode("latin-1") for k, v in self.vocab.items()},
                "special_tokens": SPECIAL_TOKENS,
            }, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        with open(path, encoding="utf-8", errors="surrogateescape") as f:
            data = json.load(f)
        tok = cls()
        tok.merges = [tuple(m) for m in data["merges"]]
        tok.vocab = {int(k): v.encode("latin-1") for k, v in data["vocab"].items()}
        tok.token2id = {v: k for k, v in tok.vocab.items()}
        tok._trained = True
        return tok


def normalizar(texto: str) -> str:
    """
    Normaliza para comparar sin que los acentos rompan la búsqueda.
    Ojo: esto es para el RETRIEVAL, no para entrenar. Al modelo le damos el
    texto tal cual, con sus tildes, porque las tildes cambian el significado
    ("público" / "publico" / "publicó").
    """
    texto = texto.lower().strip()
    descompuesto = unicodedata.normalize("NFD", texto)
    return "".join(c for c in descompuesto if unicodedata.category(c) != "Mn")
