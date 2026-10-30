"""
Base de conocimiento: el esquema de datos y su carga.

Diseño deliberado: el conocimiento NO vive en los pesos de la red neuronal,
vive aquí, en texto plano legible y editable. Con un modelo de 16M parámetros
no hay manera de que sepa nada fiable por sí solo; con esta base sí.

Consecuencias buenas de hacerlo así:
  * Se puede auditar. Abres el archivo y ves lo que sabe.
  * Se puede corregir sin reentrenar.
  * Se puede citar la fuente de cada afirmación.
  * Cada hecho lleva una CERTEZA, que es lo que alimenta la puerta de ignorancia.
  * El formato es portátil: sirve para retrieval y para generar datos de
    entrenamiento del modelo neural.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator, Optional

RUTA_CONOCIMIENTO = Path(__file__).parent / "data" / "conocimiento"


@dataclass
class Referencia:
    """
    De dónde sale un dato.

    El campo `verificado` es el importante y conviene ser explícito sobre qué
    significa. NO quiere decir "esto es verdad": quiere decir "un humano abrió
    la fuente y comprobó que el dato está ahí". Son cosas distintas, y confundirlas
    es exactamente el error que hace que un sistema de citas dé falsa seguridad.

    Un hecho sin referencias puede ser perfectamente correcto. Pero el sistema
    debe poder distinguir "tengo fuente" de "me lo sé porque lo escribí yo", y
    decirlo. Si no, las citas son decoración.
    """
    titulo: str
    url: str = ""
    medio: str = ""                 # wikipedia | libro | web | informe | ...
    consultado: str = ""            # fecha ISO en que se comprobó
    verificado: bool = False        # ¿alguien abrió la fuente y comprobó el dato?
    nota: str = ""

    def cita_corta(self) -> str:
        if self.medio and self.titulo:
            return f"{self.medio}: {self.titulo}"
        return self.titulo or self.url or "fuente sin identificar"

    @classmethod
    def from_dict(cls, d: dict) -> "Referencia":
        campos = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in campos})


@dataclass
class Hecho:
    """Una unidad de conocimiento."""
    id: str
    tema: str                              # ruta jerárquica: "geografia/capitales"
    pregunta: str                          # la pregunta canónica que responde
    respuesta: str                         # respuesta en prosa, lista para decir
    hechos: list[str] = field(default_factory=list)   # afirmaciones atómicas
    certeza: float = 0.9                   # 0..1 — cuánto nos fiamos nosotros
    nivel: str = "basico"                  # basico | medio | avanzado | meta
    fuente: str = "curado"                 # curado | conversacion | sueno
    nota: Optional[str] = None             # advertencia ("cifra aproximada", etc.)
    referencias: list[Referencia] = field(default_factory=list)
    verificado: bool = False               # ¿alguna referencia está comprobada?

    # campos que se rellenan al cargar
    archivo: Optional[str] = None
    linea: Optional[int] = None

    def raiz_tema(self) -> str:
        return self.tema.split("/")[0]

    @property
    def tiene_fuente(self) -> bool:
        return bool(self.referencias)

    @property
    def fuente_verificada(self) -> bool:
        return any(r.verificado for r in self.referencias)

    def texto_completo(self) -> str:
        """Todo el contenido indexable de este hecho."""
        partes = [self.pregunta, self.respuesta, " ".join(self.hechos)]
        if self.nota:
            partes.append(self.nota)
        return " ".join(partes)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["referencias"] = [asdict(r) for r in self.referencias]
        return {k: v for k, v in d.items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict) -> "Hecho":
        campos = cls.__dataclass_fields__
        kwargs = {k: v for k, v in d.items() if k in campos}
        refs = kwargs.pop("referencias", None) or []
        kwargs["referencias"] = [
            r if isinstance(r, Referencia) else Referencia.from_dict(r)
            for r in refs
        ]
        return cls(**kwargs)


class BaseConocimiento:
    """Contenedor de hechos con índices básicos por id y por tema."""

    def __init__(self, hechos: Optional[list[Hecho]] = None) -> None:
        self._hechos: list[Hecho] = list(hechos or [])
        self._por_id: dict[str, Hecho] = {}
        self._por_tema: dict[str, list[Hecho]] = {}
        self._reindexar()

    def _reindexar(self) -> None:
        self._por_id = {h.id: h for h in self._hechos}
        self._por_tema = {}
        for h in self._hechos:
            self._por_tema.setdefault(h.tema, []).append(h)
            self._por_tema.setdefault(h.raiz_tema(), []).append(h)

    # ------------------------------------------------------------------ #
    def añadir(self, hecho: Hecho) -> None:
        self._hechos.append(hecho)
        self._reindexar()

    def __len__(self) -> int:
        return len(self._hechos)

    def __iter__(self) -> Iterator[Hecho]:
        return iter(self._hechos)

    def __getitem__(self, idx) -> Hecho:
        return self._hechos[idx]

    def por_id(self, id_: str) -> Optional[Hecho]:
        return self._por_id.get(id_)

    def por_tema(self, tema: str) -> list[Hecho]:
        return list(self._por_tema.get(tema, []))

    def temas(self) -> list[str]:
        return sorted({h.tema for h in self._hechos})

    # ------------------------------------------------------------------ #
    @classmethod
    def cargar(cls, ruta: str | Path = RUTA_CONOCIMIENTO,
               solo_temas: Optional[list[str]] = None) -> "BaseConocimiento":
        """Carga todos los .jsonl del directorio, tolerando líneas mal formadas."""
        ruta = Path(ruta)
        hechos: list[Hecho] = []
        errores: list[str] = []

        archivos = sorted(ruta.glob("*.jsonl")) if ruta.is_dir() else [ruta]
        for arch in archivos:
            for n, linea in enumerate(arch.read_text(encoding="utf-8").splitlines(), 1):
                linea = linea.strip()
                if not linea or linea.startswith("//"):
                    continue
                try:
                    d = json.loads(linea)
                    h = Hecho.from_dict(d)
                    h.archivo, h.linea = arch.name, n
                    if solo_temas and h.raiz_tema() not in solo_temas:
                        continue
                    hechos.append(h)
                except Exception as e:                      # noqa: BLE001
                    errores.append(f"{arch.name}:{n}: {type(e).__name__}: {e}")

        base = cls(hechos)
        base.errores_carga = errores                          # type: ignore[attr-defined]
        return base

    # ------------------------------------------------------------------ #
    def estadisticas(self) -> dict:
        from collections import Counter
        por_tema = Counter(h.raiz_tema() for h in self._hechos)
        por_nivel = Counter(h.nivel for h in self._hechos)
        por_fuente = Counter(h.fuente for h in self._hechos)
        certezas = [h.certeza for h in self._hechos]
        con_ref = sum(1 for h in self._hechos if h.tiene_fuente)
        verif = sum(1 for h in self._hechos if h.fuente_verificada)
        return {
            "total_hechos": len(self._hechos),
            "temas_distintos": len({h.tema for h in self._hechos}),
            "por_tema": dict(por_tema.most_common()),
            "por_nivel": dict(por_nivel.most_common()),
            "por_fuente": dict(por_fuente.most_common()),
            "certeza_media": round(sum(certezas) / max(len(certezas), 1), 3),
            "certeza_min": min(certezas, default=0),
            "palabras_totales": sum(len(h.texto_completo().split()) for h in self._hechos),
            # Los dos que de verdad dicen si el sistema puede respaldarse.
            "con_referencia": con_ref,
            "referencia_verificada": verif,
            "cobertura_fuentes": f"{100*con_ref/max(len(self._hechos),1):.0f} %",
        }


# ---------------------------------------------------------------------------
# Escritura: así es como la IA "aprende mientras habla" y como consolida sueños
# ---------------------------------------------------------------------------
def guardar_hecho(hecho: Hecho, ruta: str | Path = RUTA_CONOCIMIENTO) -> Path:
    """Añade un hecho al final del archivo jsonl de su tema."""
    ruta = Path(ruta)
    ruta.mkdir(parents=True, exist_ok=True)
    nombre = hecho.tema.replace("/", "_") + ".jsonl"
    arch = ruta / nombre
    with open(arch, "a", encoding="utf-8") as f:
        f.write(json.dumps(hecho.to_dict(), ensure_ascii=False) + "\n")
    return arch


if __name__ == "__main__":
    base = BaseConocimiento.cargar()
    print(f"Cargados {len(base)} hechos desde {RUTA_CONOCIMIENTO}\n")
    for k, v in base.estadisticas().items():
        print(f"  {k:18} {v}")
    if getattr(base, "errores_carga", []):
        print("\nERRORES DE CARGA:")
        for e in base.errores_carga:
            print("  ", e)
    print("\nTemas:")
    for t in base.temas():
        print(f"   - {t} ({len(base.por_tema(t))} hechos)")
