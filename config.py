"""
Configuración del modelo.

La idea central del proyecto: LA ARQUITECTURA es la de un modelo grande
(DeepSeek-V3: RoPE + RMSNorm + SwiGLU + GQA + MoE con balanceo sin loss
auxiliar), pero EL TAMAÑO es de juguete, para que entre y entrerene en una
máquina sin GPU.

Cambiar de `toy` a `medium` no toca ni una línea del modelo: solo números
de aquí. Esa es la gracia de hacerlo bien desde el principio.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional


@dataclass
class ModelConfig:
    # --- vocabulario / forma ---
    vocab_size: int = 4096          # lo fija el tokenizer BPE que entrenemos
    d_model: int = 256              # ancho del embedding residual
    n_layers: int = 8
    max_seq_len: int = 256

    # --- atención (GQA: menos cabezas K/V que Q, como DeepSeek-V2/V3) ---
    n_heads: int = 8                # cabezas de query
    n_kv_heads: int = 2             # cabezas de key/value compartidas -> 4x menos KV
    rope_theta: float = 10_000.0

    # --- FFN ---
    ffn_multiple: float = 2.6667    # SwiGLU usa 3 matrices; 8/3 mantiene FLOPs
    ffn_hidden: Optional[int] = None  # si None se calcula desde ffn_multiple

    # --- Mixture of Experts ---
    # Detalle clave de DeepSeekMoE: los expertos son ESTRECHOS (ffn_hidden pequeño)
    # y muchos. Así caben en memoria y encima solo se activan top_k por token.
    # Si pusiéramos expertos del tamaño del FFN denso, el modelo explotaría.
    n_experts: int = 8              # expertos ruteados por capa
    top_k: int = 2                  # cuántos se activan por token
    shared_experts: int = 1         # expertos siempre activos (DeepSeekMoE)
    moe_ffn_hidden: Optional[int] = None     # ancho de cada experto ruteado
    shared_ffn_hidden: Optional[int] = None  # ancho del experto compartido
    use_moe: bool = True            # False -> FFN denso clásico
    moe_layer_freq: int = 1         # cada cuántas capas hay MoE

    # --- regularización / init ---
    dropout: float = 0.0
    init_std: float = 0.02
    tie_embeddings: bool = True     # compartir embedding de entrada y salida

    # --- balanceo sin loss auxiliar (DeepSeek-V3) ---
    # En lugar de penalizar el desbalance en la pérdida (que degrada el modelo),
    # se añaden "bias" a los logits del router: al experto más usado se le baja
    # la puntuación y al menos usado se le sube. El balance emerge de ahí.
    aux_loss_free: bool = True
    balance_bias_lr: float = 1e-2   # paso de actualización de esos bias
    balance_target: Optional[float] = None  # None -> 1/n_experts

    def __post_init__(self) -> None:
        if self.ffn_hidden is None:
            # múltiplo de 16 para que las matrices queden alineadas
            self.ffn_hidden = int(self.d_model * self.ffn_multiple // 16) * 16
        if self.moe_ffn_hidden is None:
            # expertos ruteados: la cuarta parte del FFN denso
            self.moe_ffn_hidden = max(16, int(self.ffn_hidden // 4 // 16) * 16)
        if self.shared_ffn_hidden is None:
            self.shared_ffn_hidden = self.ffn_hidden
        if self.balance_target is None:
            self.balance_target = 1.0 / self.n_experts
        assert self.d_model % self.n_heads == 0, "d_model debe dividir a n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads debe dividir a n_kv_heads"
        assert self.top_k <= self.n_experts, "top_k no puede superar a n_experts"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def kv_dim(self) -> int:
        return self.head_dim * self.n_kv_heads

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# Presets: misma arquitectura, distinto tamaño.
# ---------------------------------------------------------------------------
PRESETS: dict[str, dict] = {
    # --- Los dos primeros caben y entrenan en ESTE sandbox (2 CPU, 1 GB RAM) ---

    # ~7M parámetros. Para validar el pipeline completo en minutos:
    # forward, backward, AdamW, MoE, y que la pérdida baje de verdad.
    "tiny": dict(
        vocab_size=4096, d_model=256, n_layers=6, max_seq_len=256,
        n_heads=8, n_kv_heads=2,
        ffn_hidden=512, moe_ffn_hidden=128, shared_ffn_hidden=256,
        n_experts=8, top_k=2, shared_experts=1,
    ),

    # ~14M. El modelo "de producto" para esta máquina.
    "toy": dict(
        vocab_size=4096, d_model=256, n_layers=8, max_seq_len=256,
        n_heads=8, n_kv_heads=2,
        ffn_hidden=512, moe_ffn_hidden=256, shared_ffn_hidden=256,
        n_experts=8, top_k=2, shared_experts=1,
    ),

    # Sin MoE. Sirve para comparar: ¿el MoE ayuda o solo añade complejidad?
    "toy_dense": dict(
        vocab_size=4096, d_model=256, n_layers=8, max_seq_len=256,
        n_heads=8, n_kv_heads=2, ffn_hidden=512, use_moe=False,
    ),

    # --- De aquí en adelante necesitan otra máquina (los dejamos definidos
    #     para que escalar sea cambiar una palabra, no reescribir el modelo) ---

    # ~55M. Un portátil con 16 GB de RAM.
    "small": dict(
        vocab_size=8192, d_model=512, n_layers=12, max_seq_len=512,
        n_heads=8, n_kv_heads=2,
        moe_ffn_hidden=256, shared_ffn_hidden=512,
        n_experts=16, top_k=2,
    ),

    # ~350M. Una GPU alquilada. A partir de aquí el modelo ya empieza a
    # saber cosas por sí solo, sin depender tanto del retrieval.
    "medium": dict(
        vocab_size=16384, d_model=1024, n_layers=24, max_seq_len=1024,
        n_heads=16, n_kv_heads=4,
        moe_ffn_hidden=512, shared_ffn_hidden=1024,
        n_experts=32, top_k=2,
    ),
}


def get_config(name: str = "toy", **overrides) -> ModelConfig:
    """Devuelve una config por nombre de preset, con overrides opcionales."""
    if name not in PRESETS:
        raise KeyError(f"preset desconocido {name!r}. Disponibles: {list(PRESETS)}")
    params = {**PRESETS[name], **overrides}
    return ModelConfig(**params)


def estimate_params(cfg: ModelConfig) -> dict:
    """
    Cuenta parámetros totales y ACTIVOS por token sin construir el modelo.
    En un MoE la diferencia es lo que hace que sea barato de correr:
    DeepSeek-V3 tiene 671B totales pero solo activa 37B por token.
    """
    d, L, V = cfg.d_model, cfg.n_layers, cfg.vocab_size
    kv = cfg.kv_dim
    h = cfg.ffn_hidden
    he = cfg.moe_ffn_hidden
    hs = cfg.shared_ffn_hidden

    emb = V * d
    out = 0 if cfg.tie_embeddings else V * d

    # atención: q, k, v, o
    attn = d * d + 2 * d * kv + kv * d

    ffn_dense = 3 * d * h                      # gate, up, down (SwiGLU)
    router = d * cfg.n_experts
    expert = 3 * d * he                        # un experto ruteado
    moe = router + cfg.n_experts * expert + cfg.shared_experts * 3 * d * hs
    moe_act = router + cfg.top_k * expert + cfg.shared_experts * 3 * d * hs

    ffn_per_layer = moe if cfg.use_moe else ffn_dense
    ffn_act_layer = moe_act if cfg.use_moe else ffn_dense

    n_moe_layers = len([i for i in range(L) if (i + 1) % cfg.moe_layer_freq == 0]) if cfg.use_moe else 0
    n_dense_layers = L - n_moe_layers

    per_layer = attn + ffn_dense   # capas densas
    per_moe = attn + moe           # capas MoE

    total = emb + out + n_dense_layers * per_layer + n_moe_layers * per_moe + d
    total_act = emb + out + L * (attn + ffn_act_layer) + d

    return {
        "totales": int(total),
        "activos_por_token": int(total_act),
        "embedding": int(emb + out),
        "atencion_total": int(attn * L),
        "ffn_total": int(n_dense_layers * ffn_dense + n_moe_layers * moe),
        "capas_moe": n_moe_layers,
        "capas_densas": n_dense_layers,
    }


if __name__ == "__main__":
    import os

    def ram_mb() -> float:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable"):
                        return int(line.split()[1]) / 1024
        except OSError:
            pass
        return float("nan")

    disponible = ram_mb()
    print(f"RAM disponible en esta máquina: {disponible:.0f} MB\n")

    for name in PRESETS:
        cfg = get_config(name)
        p = estimate_params(cfg)
        # pesos + gradientes + 2 momentos de Adam, todo en fp32
        coste = p["totales"] * 16 / 1e6
        cabe = "SÍ" if coste < disponible * 0.5 else "NO"
        print(f"=== {name} ===")
        print(f"  d_model={cfg.d_model}  capas={cfg.n_layers}  "
              f"heads={cfg.n_heads}q/{cfg.n_kv_heads}kv  ctx={cfg.max_seq_len}")
        if cfg.use_moe:
            print(f"  MoE: {cfg.n_experts} expertos (h={cfg.moe_ffn_hidden}), "
                  f"top-{cfg.top_k}, +{cfg.shared_experts} compartido "
                  f"(h={cfg.shared_ffn_hidden})")
        else:
            print(f"  FFN denso h={cfg.ffn_hidden}")
        print(f"  parámetros totales   : {p['totales']:>12,}")
        print(f"  activos por token    : {p['activos_por_token']:>12,}"
              f"   ({100*p['activos_por_token']/p['totales']:.0f}%)")
        print(f"  coste entrenamiento  : {coste:>9.0f} MB  -> ¿cabe aquí? {cabe}")
        print()
