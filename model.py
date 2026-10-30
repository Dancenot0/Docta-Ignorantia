"""
El modelo: un Transformer con la arquitectura de DeepSeek, en miniatura.

Todo escrito a mano sobre NumPy. No hay autograd: el backward de cada pieza
está derivado y programado explícitamente. Es más trabajo, pero es la única
manera de saber de verdad qué está pasando dentro, y además nos obliga a
entender cada pieza en vez de confiar en una caja negra.

Piezas, y por qué cada una está ahí:

  RMSNorm     Normaliza sin centrar (solo escala). Más barata que LayerNorm y
              empíricamente igual de buena. DeepSeek, Llama y compañía la usan.
              Se aplica ANTES de cada subcapa (norma pre-activación), que es lo
              que hace que entrenar redes profundas sea estable.

  RoPE        Codifica posiciones girando los vectores de query y key. La gracia
              es que el producto escalar entre dos tokens pasa a depender solo de
              la DISTANCIA entre ellos, no de su posición absoluta. Generaliza a
              secuencias más largas de las vistas en entrenamiento.

  GQA         Varias cabezas de query comparten las mismas key/value. La cache de
              atención durante la generación es proporcional a las cabezas KV,
              así que esto ahorra mucha memoria casi sin perder calidad.

  SwiGLU      FFN con tres matrices y una puerta multiplicativa:
              down( silu(gate(x)) * up(x) ). Es el FFN estándar hoy.

  MoE         En vez de un FFN grande, muchos FFN pequeños ("expertos") y un
              router que elige top-k por token. Solo se calculan esos k, así que
              el coste por token es el de un FFN pequeño aunque el modelo tenga
              muchísimos parámetros. Es lo que lo hace barato de USAR.

  Balanceo sin loss auxiliar
              El problema clásico del MoE es que el router se enamora de dos o
              tres expertos y los demás no aprenden nunca. La solución vieja era
              añadir a la pérdida un término que penalizara el desequilibrio, lo
              cual distorsiona el objetivo real. DeepSeek-V3 hace otra cosa:
              mantener un "bias" por experto, restárselo a los logits del router,
              y ajustar ese bias con un controlador proporcional propio, FUERA
              del descenso de gradiente. Al experto más usado se le baja la
              puntuación y al menos usado se le sube. El equilibrio emerge sin
              ensuciar la pérdida. Está en `_balance_update`.

Qué NO tiene, a propósito:
  * Multi-head Latent Attention (MLA), la compresión de KV de DeepSeek-V2/V3.
    Es una optimización de memoria para inferencia larga; con ctx=256 no cambia
    nada observable y complica el backward muchísimo.
  * Multi-Token Prediction. Útil a escala, ruido a esta escala.
  Quedan documentados como siguiente paso.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

from config import ModelConfig


# ===========================================================================
# Piezas sueltas
# ===========================================================================
def rmsnorm(x: np.ndarray, w: np.ndarray, eps: float = 1e-6
            ) -> tuple[np.ndarray, np.ndarray]:
    """y = (x / rms(x)) * w. Devuelve también rms, que el backward necesita."""
    ms = np.mean(x * x, axis=-1, keepdims=True)
    rms = np.sqrt(ms + eps)
    return (x / rms) * w, rms


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softmax_masked(scores: np.ndarray, mask_add: np.ndarray) -> np.ndarray:
    """Softmax en el último eje con máscara aditiva (-inf donde no se atiende)."""
    s = scores + mask_add
    s = s - s.max(axis=-1, keepdims=True)          # estable numéricamente
    e = np.exp(s)
    return e / e.sum(axis=-1, keepdims=True)


def cross_entropy(logits: np.ndarray, targets: np.ndarray, mask: np.ndarray
                  ) -> tuple[float, np.ndarray]:
    """
    Pérdida de siguiente-token. `targets` ya viene DESPLAZADO: targets[b,t] es
    el token que debe predecir logits[b,t]. mask=1 donde la predicción cuenta.

    El gradiente de la entropía cruzada con softmax es precioso:
        dL/dlogits = probs - onehot(target)
    escalado por la máscara y dividido entre el número de posiciones válidas.
    """
    m = logits.max(axis=-1, keepdims=True)
    z = logits - m
    exp_z = np.exp(z)
    sumexp = exp_z.sum(axis=-1, keepdims=True)
    logsumexp = np.log(sumexp) + m

    logit_target = np.take_along_axis(z, targets[..., None], axis=-1) + m
    nll = (logsumexp - logit_target)[..., 0]

    denom = max(float(mask.sum()), 1.0)
    loss = float((nll * mask).sum() / denom)

    probs = exp_z / sumexp                          # (B,T,V)
    dlogits = probs * (mask / denom)[..., None]
    # restamos 1·mask/denom en la posición del target: (probs − onehot) escalado.
    # Se hace in-place solo en una columna para no construir el one-hot entero,
    # que sería una matriz (B,T,V) más y con V grande duele.
    en_target = np.take_along_axis(dlogits, targets[..., None], axis=-1)
    np.put_along_axis(dlogits, targets[..., None],
                      en_target - (mask / denom)[..., None], axis=-1)
    return loss, dlogits


# ===========================================================================
# RoPE
# ===========================================================================
class RopeTablas:
    """Cosenos y senos precalculados. Sin parámetros entrenables."""

    def __init__(self, head_dim: int, max_len: int, theta: float = 10_000.0,
                 dtype=np.float32):
        # Frecuencias theta^(-2i/d): las primeras dimensiones rotan rápido
        # (posición fina) y las últimas lentísimo (posición gruesa). Es un reloj
        # multiescala, y por eso capta tanto el token vecino como el párrafo.
        i = np.arange(0, head_dim, 2, dtype=np.float64)
        inv_freq = theta ** (-i / head_dim)
        ang = np.outer(np.arange(max_len, dtype=np.float64), inv_freq)
        self.cos = np.cos(ang).astype(dtype)          # (max_len, head_dim/2)
        self.sin = np.sin(ang).astype(dtype)

    def _tablas(self, ndim: int, T: int, hd: int):
        shape = (1,) * (ndim - 2) + (T, hd // 2)
        return self.cos[:T].reshape(shape), self.sin[:T].reshape(shape)

    def aplicar(self, x: np.ndarray, cache: dict, clave: str) -> np.ndarray:
        """Gira cada par (2i, 2i+1). Guarda lo necesario para el backward."""
        T, hd = x.shape[-2], x.shape[-1]
        cos, sin = self._tablas(x.ndim, T, hd)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        out = np.empty_like(x)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        cache[clave] = (cos, sin)
        return out

    def retro(self, dy: np.ndarray, cache: dict, clave: str) -> np.ndarray:
        """El backward de una rotación es la rotación inversa."""
        cos, sin = cache[clave]
        dy1, dy2 = dy[..., 0::2], dy[..., 1::2]
        dx = np.empty_like(dy)
        dx[..., 0::2] = dy1 * cos + dy2 * sin
        dx[..., 1::2] = -dy1 * sin + dy2 * cos
        return dx


# ===========================================================================
# El modelo
# ===========================================================================
class Modelo:
    """
    Forward y backward completos.

    Los parámetros viven en un dict plano {nombre: array}. Eso hace trivial
    contarlos, guardarlos y —lo más importante— hacer comprobación numérica de
    gradientes índice a índice, que es la única prueba de que el backward está
    bien escrito.
    """

    def __init__(self, cfg: ModelConfig, semilla: int = 1234,
                 dtype=np.float32):
        """
        dtype importa más de lo que parece. Entrenamos en float32 porque es la
        mitad de memoria, pero la COMPROBACIÓN de gradientes se hace en
        float64: en float32 la pérdida está cuantizada (el salto mínimo entre
        dos valores representables ronda 1e-7) y una perturbación pequeña no
        llega a cambiarla, así que el gradiente numérico sale 0 y parece que
        el backward está roto cuando no lo está.
        """
        self.cfg = cfg
        self.dtype = np.dtype(dtype)
        rng = np.random.default_rng(semilla)
        d = cfg.d_model
        self.p: dict[str, np.ndarray] = {}
        self.g: dict[str, np.ndarray] = {}

        # Escala de init: al residual se suman n_layers aportaciones, así que
        # cada una debe ir dividida por sqrt(2*n_layers) para que la varianza no
        # explote al atravesar la red. Es lo que hace que una red honda arranque.
        esc_attn = cfg.init_std * math.sqrt(1.0 / max(2 * cfg.n_layers, 1))

        def nuevo(nombre: str, forma: tuple, std: Optional[float] = None) -> np.ndarray:
            w = (rng.standard_normal(forma) * (std if std is not None else cfg.init_std)
                 ).astype(self.dtype)
            self.p[nombre] = w
            self.g[nombre] = np.zeros_like(w)
            return w

        def unos(nombre: str, forma: tuple) -> None:
            """Las normas se inicializan en 1. En ~0 anularían la señal y la red
            tardaría muchísimo en arrancar."""
            self.p[nombre] = np.ones(forma, self.dtype)
            self.g[nombre] = np.zeros(forma, self.dtype)

        nuevo("emb.w", (cfg.vocab_size, d), std=d ** -0.5)
        if not cfg.tie_embeddings:
            nuevo("out.w", (cfg.vocab_size, d), std=d ** -0.5)
        unos("fn.w", (d,))

        self.es_moe = [cfg.use_moe and ((l + 1) % cfg.moe_layer_freq == 0)
                       for l in range(cfg.n_layers)]

        for l in range(cfg.n_layers):
            pre = f"l{l}"
            unos(f"{pre}.an.w", (d,))
            unos(f"{pre}.fn.w", (d,))

            nuevo(f"{pre}.wq", (d, d), std=esc_attn)
            nuevo(f"{pre}.wk", (d, cfg.kv_dim), std=esc_attn)
            nuevo(f"{pre}.wv", (d, cfg.kv_dim), std=esc_attn)
            nuevo(f"{pre}.wo", (d, d), std=esc_attn)

            if self.es_moe[l]:
                he, hs = cfg.moe_ffn_hidden, cfg.shared_ffn_hidden
                nuevo(f"{pre}.router", (d, cfg.n_experts), std=0.01)
                for e in range(cfg.n_experts):
                    nuevo(f"{pre}.e{e}.wg", (d, he))
                    nuevo(f"{pre}.e{e}.wu", (d, he))
                    nuevo(f"{pre}.e{e}.wd", (he, d))
                for s in range(cfg.shared_experts):
                    nuevo(f"{pre}.s{s}.wg", (d, hs))
                    nuevo(f"{pre}.s{s}.wu", (d, hs))
                    nuevo(f"{pre}.s{s}.wd", (hs, d))
            else:
                h = cfg.ffn_hidden
                nuevo(f"{pre}.wg", (d, h))
                nuevo(f"{pre}.wu", (d, h))
                nuevo(f"{pre}.wd", (h, d))

        # bias de balanceo: NO se entrenan con Adam, van con regla propia.
        self._usa_balance = bool(cfg.use_moe and cfg.aux_loss_free)
        if self._usa_balance:
            for l in range(cfg.n_layers):
                if self.es_moe[l]:
                    self.p[f"l{l}.balance_bias"] = np.zeros(cfg.n_experts, self.dtype)
                    self.g[f"l{l}.balance_bias"] = np.zeros(cfg.n_experts, self.dtype)

        self.rope = RopeTablas(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta,
                               dtype=self.dtype)
        self._C: dict = {}
        self._inyectar: dict = {}

    # ------------------------------------------------------------------ #
    @property
    def params_entrenables(self) -> list[str]:
        """Todo menos los bias de balanceo, que tienen su propia regla."""
        return [k for k in self.p if not k.endswith("balance_bias")]

    def n_params(self) -> int:
        return sum(self.p[k].size for k in self.params_entrenables)

    # ================================================================== #
    # FORWARD
    # ================================================================== #
    def forward(self, ids: np.ndarray, mask_loss: Optional[np.ndarray] = None
                ) -> tuple[float, np.ndarray]:
        cfg = self.cfg
        C: dict = self._C
        C.clear()

        B, T = ids.shape
        assert T <= cfg.max_seq_len, f"secuencia de {T} tokens > ctx {cfg.max_seq_len}"
        if mask_loss is not None:
            # Aserción defensiva a propósito: si las formas no encajan, que
            # reviente AQUÍ con un mensaje que se entienda, y no dentro de
            # cross_entropy con un error de broadcasting críptico.
            assert mask_loss.shape == (B, T - 1), (
                f"la máscara debe tener forma (B, T-1) = {(B, T - 1)} porque cada "
                f"posición predice el token siguiente; se recibió {mask_loss.shape}")
        C["ids"], C["B"], C["T"] = ids, B, T

        x = self.p["emb.w"][ids].astype(self.dtype)          # (B,T,d)

        for l in range(cfg.n_layers):
            pre = f"l{l}"
            C[f"{pre}.x_in"] = x

            # ---------------- atención ----------------
            xn, rms_a = rmsnorm(x, self.p[f"{pre}.an.w"])
            C[f"{pre}.xn"], C[f"{pre}.rms_a"] = xn, rms_a

            q = xn @ self.p[f"{pre}.wq"]
            k = xn @ self.p[f"{pre}.wk"]
            v = xn @ self.p[f"{pre}.wv"]

            hd, nh, nkv = cfg.head_dim, cfg.n_heads, cfg.n_kv_heads
            qh = q.reshape(B, T, nh, hd).transpose(0, 2, 1, 3)     # (B,nh,T,hd)
            kh = k.reshape(B, T, nkv, hd).transpose(0, 2, 1, 3)
            vh = v.reshape(B, T, nkv, hd).transpose(0, 2, 1, 3)

            qr = self.rope.aplicar(qh, C, f"{pre}.ropeq")
            kr = self.rope.aplicar(kh, C, f"{pre}.ropek")
            C[f"{pre}.qh"], C[f"{pre}.kh"], C[f"{pre}.vh"] = qh, kh, vh
            C[f"{pre}.qr"], C[f"{pre}.kr"] = qr, kr

            rep = nh // nkv
            kr_r = np.repeat(kr, rep, axis=1) if rep > 1 else kr
            vh_r = np.repeat(vh, rep, axis=1) if rep > 1 else vh
            C[f"{pre}.kr_r"], C[f"{pre}.vh_r"], C[f"{pre}.rep"] = kr_r, vh_r, rep

            esc = self.dtype.type(1.0 / math.sqrt(hd))
            C[f"{pre}.esc"] = esc
            s_att = (qr @ kr_r.transpose(0, 1, 3, 2)) * esc        # (B,nh,T,T)

            causal = np.triu(np.full((T, T), -1e9, self.dtype), k=1)
            C[f"{pre}.causal"] = causal
            att = softmax_masked(s_att, causal[None, None])
            C[f"{pre}.att"] = att

            ctx = att @ vh_r                                       # (B,nh,T,hd)
            C[f"{pre}.ctx"] = ctx
            ctx_flat = ctx.transpose(0, 2, 1, 3).reshape(B, T, nh * hd)
            C[f"{pre}.ctx_flat"] = ctx_flat

            o = ctx_flat @ self.p[f"{pre}.wo"]
            if self._inyectar:
                o = o + self._inyectar.get(f"{pre}.attn_out", 0.0)
            C[f"{pre}.attn_out"] = o
            x = x + o
            C[f"{pre}.x_mid"] = x

            # ---------------- FFN / MoE ----------------
            xn2, rms_f = rmsnorm(x, self.p[f"{pre}.fn.w"])
            if self._inyectar:                       # gancho de diagnóstico
                xn2 = xn2 + self._inyectar.get(f"{pre}.xn2", 0.0)
            C[f"{pre}.xn2"], C[f"{pre}.rms_f"] = xn2, rms_f

            if self.es_moe[l]:
                f = self._moe_fwd(pre, xn2, C)
            else:
                f = self._ffn_fwd(pre, xn2, C)
            # Gancho de diagnóstico: permite perturbar un tensor intermedio desde
            # fuera para comprobar numéricamente el gradiente del flujo residual.
            # En operación normal `_inyectar` está vacío y esto no hace nada.
            if self._inyectar:
                f = f + self._inyectar.get(f"{pre}.ffn_out", 0.0)
            C[f"{pre}.ffn_out"] = f
            x = x + f

        C["x_fin"] = x
        hf, rms_fin = rmsnorm(x, self.p["fn.w"])
        C["hf"], C["rms_fin"] = hf, rms_fin

        w_out = self.p["emb.w"] if cfg.tie_embeddings else self.p["out.w"]
        logits = hf @ w_out.T
        C["logits"] = logits

        pred, targ = logits[:, :-1, :], ids[:, 1:]
        if mask_loss is None:
            mask_loss = np.ones_like(targ, dtype=self.dtype)
        mask_loss = mask_loss.astype(self.dtype)
        C["mask_loss"], C["targ"] = mask_loss, targ

        loss, d_pred = cross_entropy(pred, targ, mask_loss)
        dlogits = np.zeros_like(logits)
        dlogits[:, :-1, :] = d_pred
        C["dlogits"] = dlogits
        return loss, logits

    # ---------------- FFN denso ----------------
    def _ffn_fwd(self, pre: str, x: np.ndarray, C: dict) -> np.ndarray:
        # Todo se cachea aplanado a (N, ·). El backward son multiplicaciones de
        # matrices sobre tokens, y mezclar 3D con 2D es una fuente de errores de
        # broadcasting que encima a veces "funcionan" dando números falsos.
        x2 = x.reshape(-1, x.shape[-1])
        g = x2 @ self.p[f"{pre}.wg"]
        u = x2 @ self.p[f"{pre}.wu"]
        sg = silu(g)
        a = sg * u
        C[f"{pre}.g"], C[f"{pre}.u"], C[f"{pre}.sg"], C[f"{pre}.a"] = g, u, sg, a
        out = a @ self.p[f"{pre}.wd"]
        return out.reshape(x.shape)

    # ---------------- MoE ----------------
    def _moe_fwd(self, pre: str, x: np.ndarray, C: dict) -> np.ndarray:
        cfg = self.cfg
        B, T, d = x.shape
        flat = x.reshape(-1, d)                                   # (N,d)
        N = flat.shape[0]
        C[f"{pre}.flat"] = flat

        rl = flat @ self.p[f"{pre}.router"]                       # (N,E)
        rl_adj = rl - self.p[f"{pre}.balance_bias"] if self._usa_balance else rl
        C[f"{pre}.rl_adj"] = rl_adj

        sig = sigmoid(rl_adj)
        C[f"{pre}.sig"] = sig

        k = cfg.top_k
        idx = np.argpartition(-rl_adj, k - 1, axis=1)[:, :k]
        # argpartition no ordena dentro del top-k; ordenamos para que sea
        # determinista y el top-1 sea siempre el índice 0.
        vals = np.take_along_axis(rl_adj, idx, axis=1)
        orden = np.argsort(-vals, axis=1)
        idx = np.take_along_axis(idx, orden, axis=1)
        C[f"{pre}.idx"] = idx

        g_sel = np.take_along_axis(sig, idx, axis=1)
        suma = g_sel.sum(axis=1, keepdims=True) + 1e-9
        g_norm = g_sel / suma
        C[f"{pre}.g_sel"], C[f"{pre}.g_norm"], C[f"{pre}.g_sum"] = g_sel, g_norm, suma

        out = np.zeros((N, d), self.dtype)
        C[f"{pre}.exp_a"] = []
        for j in range(k):
            e = idx[:, j]
            a = self._experto_fwd(f"{pre}.e", e, flat, C, j)
            C[f"{pre}.exp_a"].append(a)
            out += g_norm[:, j:j + 1] * a

        for s in range(cfg.shared_experts):
            sp = f"{pre}.s{s}"
            o = self._swiglu_fwd(sp, flat, C)
            out += o

        if self._usa_balance:
            uso = np.bincount(idx.ravel(), minlength=cfg.n_experts).astype(self.dtype)
            C[f"{pre}.uso"] = uso / max(N * k, 1)

        return out.reshape(B, T, d)

    def _swiglu_fwd(self, pref: str, x: np.ndarray, C: dict) -> np.ndarray:
        """FFN SwiGLU genérico, usado por los expertos compartidos. `x` ya llega
        aplanado a (N, d)."""
        g = x @ self.p[f"{pref}.wg"]
        u = x @ self.p[f"{pref}.wu"]
        sg = silu(g)
        a = sg * u
        C[(pref, "g")], C[(pref, "u")] = g, u
        C[(pref, "sg")], C[(pref, "a")] = sg, a
        return a @ self.p[f"{pref}.wd"]

    def _experto_fwd(self, pref: str, e: np.ndarray, flat: np.ndarray,
                     C: dict, j: int) -> np.ndarray:
        """
        Forward de un experto ruteado.

        A esta escala recorremos los expertos y seleccionamos sus tokens con una
        máscara booleana. En un modelo grande se hace al revés (gather de tokens
        por experto) para no desperdiciar nada; la lógica es la misma.
        """
        out = np.zeros_like(flat)
        for eid in np.unique(e):
            sel = e == eid
            xs = flat[sel]
            g = xs @ self.p[f"{pref}{eid}.wg"]
            u = xs @ self.p[f"{pref}{eid}.wu"]
            sg = silu(g)
            a = sg * u
            out[sel] = a @ self.p[f"{pref}{eid}.wd"]
            C[(pref, int(eid), j)] = (xs, g, u, sg, a, sel)
        return out

    # ================================================================== #
    # BACKWARD
    # ================================================================== #
    def backward(self) -> None:
        """Rellena self.g con los gradientes de la última forward()."""
        cfg = self.cfg
        C = self._C
        for v in self.g.values():
            v[...] = 0.0

        nh, nkv, hd, d = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.d_model
        B, T = C["B"], C["T"]
        _2d = lambda z: z.reshape(-1, z.shape[-1])          # noqa: E731

        dlogits = C["dlogits"]
        hf = C["hf"]
        w_out = self.p["emb.w"] if cfg.tie_embeddings else self.p["out.w"]
        clave_out = "emb.w" if cfg.tie_embeddings else "out.w"

        # ---- cabeza de salida ----
        self.g[clave_out] += _2d(dlogits).T @ _2d(hf)
        dhf = dlogits @ w_out

        # ---- RMSNorm final ----
        dx, dw_fin = self._rms_bwd(dhf, C["x_fin"], C["rms_fin"], "fn.w")
        self.g["fn.w"] += dw_fin

        # ---- capas, de la última a la primera ----
        #
        # x_in ──┬───────────────────►(+)──► x_mid ──┬──────────────►(+)──► x_out
        #        │                    ▲              │               ▲
        #        └── norm ── attn ─ wo┘              └── norm ── ffn ┘
        #
        # El orden del backward importa y es contraintuitivo. La normalización
        # NO está en el camino residual: es una rama que sale de x_mid y vuelve
        # a sumarse en x_out. Así que, bajando desde dx = dL/d(x_out):
        #
        #   1. dL/d(ffn_out) = dx, directamente, porque ffn_out se SUMA a x_mid.
        #   2. Atravesamos el FFN hacia atrás y obtenemos dL/d(xn2).
        #   3. Atravesamos la norma hacia atrás: de dL/d(xn2) salen DOS cosas,
        #      el dw de la norma y dL/d(x_mid) por esa rama.
        #   4. dL/d(x_mid) total = dx (residual) + lo del paso 3.
        #
        # El fallo típico es hacer 3 antes que 1 y darle al FFN el gradiente que
        # sale de la norma en vez del que entra por su salida. Se parecen en un
        # factor del orden de 1/rms², o sea cientos de veces, y la pérdida baja
        # igual: entrena mal en silencio. Para eso existe la comprobación
        # numérica de gradientes.
        for l in reversed(range(cfg.n_layers)):
            pre = f"l{l}"

            # ============ bloque FFN / MoE ============
            if self.es_moe[l]:
                d_ffn_out = self._moe_bwd(pre, dx, C, _2d)     # dx = dL/d(ffn_out)
            else:
                d_ffn_out = self._ffn_bwd(pre, dx, C)          # -> dL/d(xn2)

            d_rama, dw_fn = self._rms_bwd(d_ffn_out, C[f"{pre}.x_mid"],
                                          C[f"{pre}.rms_f"], f"{pre}.fn.w")
            self.g[f"{pre}.fn.w"] += dw_fn
            d_xmid = dx + d_rama                   # residual + rama del FFN

            # ============ bloque atención ============
            do = d_xmid                            # dL/d(salida de wo)
            # o = ctx_flat @ wo  ->  dwo = ctx_flat.T @ do
            # OJO al orden. Como wo SIEMPRE es cuadrada (d_model == n_heads *
            # head_dim por construcción), escribirla al revés no da error de
            # formas: da la matriz transpuesta y entrena en silencio hacia otro
            # sitio. Las otras proyecciones (wq/wk/wv/wg/wu/wd) no son cuadradas
            # y por eso ese mismo error ahí sí revienta.
            self.g[f"{pre}.wo"] += _2d(C[f"{pre}.ctx_flat"]).T @ _2d(do)
            dctx_flat = do @ self.p[f"{pre}.wo"].T

            dctx = dctx_flat.reshape(B, T, nh, hd).transpose(0, 2, 1, 3)
            att, vr = C[f"{pre}.att"], C[f"{pre}.vh_r"]

            datt = dctx @ vr.transpose(0, 1, 3, 2)          # (B,nh,T,T)
            dvr = att.transpose(0, 1, 3, 2) @ dctx          # (B,nh,T,hd)

            # softmax: dS = P * (dP - sum(dP*P))
            dS = att * (datt - (datt * att).sum(axis=-1, keepdims=True))
            dS = dS * C[f"{pre}.esc"]
            # la máscara causal pone -1e9: ahí la probabilidad es ~0 y el
            # gradiente debe ser exactamente 0, no un residuo.
            dS = np.where(C[f"{pre}.causal"][None, None] < -1e8, 0.0, dS)

            qr, kr_r = C[f"{pre}.qr"], C[f"{pre}.kr_r"]
            dqr = dS @ kr_r
            dkr_r = dS.transpose(0, 1, 3, 2) @ qr

            rep = C[f"{pre}.rep"]
            if rep > 1:                                     # deshacer el repeat de GQA
                dkr = dkr_r.reshape(B, nkv, rep, T, hd).sum(axis=2)
                dvh = dvr.reshape(B, nkv, rep, T, hd).sum(axis=2)
            else:
                dkr, dvh = dkr_r, dvr

            dqh = self.rope.retro(dqr, C, f"{pre}.ropeq")
            dkh = self.rope.retro(dkr, C, f"{pre}.ropek")

            dq = dqh.transpose(0, 2, 1, 3).reshape(B, T, nh * hd)
            dk = dkh.transpose(0, 2, 1, 3).reshape(B, T, nkv * hd)
            dv = dvh.transpose(0, 2, 1, 3).reshape(B, T, nkv * hd)

            xn_f = _2d(C[f"{pre}.xn"])
            self.g[f"{pre}.wq"] += xn_f.T @ _2d(dq)
            self.g[f"{pre}.wk"] += xn_f.T @ _2d(dk)
            self.g[f"{pre}.wv"] += xn_f.T @ _2d(dv)

            dxn = dq @ self.p[f"{pre}.wq"].T + dk @ self.p[f"{pre}.wk"].T + dv @ self.p[f"{pre}.wv"].T
            dx_in, dw_an = self._rms_bwd(dxn, C[f"{pre}.x_in"], C[f"{pre}.rms_a"],
                                         f"{pre}.an.w")
            self.g[f"{pre}.an.w"] += dw_an

            dx = dx_in + d_xmid          # residual del bloque de atención

        # ---- embedding ----
        demb = np.zeros_like(self.p["emb.w"])
        np.add.at(demb, C["ids"].ravel(), _2d(dx))
        self.g["emb.w"] += demb

        # ---- Red de seguridad: ningún gradiente puede tener forma distinta a
        # la de su peso. Si la tuviera, el += habría dado broadcasting y el
        # entrenamiento sería basura sin ningún error visible.
        for k in self.params_entrenables:
            if self.g[k].shape != self.p[k].shape:
                raise AssertionError(
                    f"gradiente de {k} con forma {self.g[k].shape} != peso {self.p[k].shape}")

        # ---- bias de balanceo (regla propia, no gradiente) ----
        if self._usa_balance:
            self._balance_update(C)

    # ---------------- backward de RMSNorm ----------------
    def _rms_bwd(self, dy: np.ndarray, x: np.ndarray, rms: np.ndarray,
                 nombre_w: str) -> tuple[np.ndarray, np.ndarray]:
        """
        y = (x/rms) * w   con   rms = sqrt(mean(x²) + eps)

        dw  = Σ_{batch,tiempo} dy·(x/rms)
        dx  = [ w·dy − x̂·mean(w·dy·x̂) ] / rms        (x̂ = x/rms)

        El segundo término de dx es el que viene de que rms depende de x: al
        escalar x también se escala rms, y eso amortigua la salida. Es el paso
        que más se olvida al derivar una normalización a mano.

        Devuelve (dx, dw) y NO toca self.g: quien llama decide dónde acumular.
        Así se puede usar para depurar sin contaminar los gradientes.
        """
        xn = x / rms
        dw = (dy * xn).sum(axis=tuple(range(dy.ndim - 1)))
        t = dy * self.p[nombre_w]
        c = (t * xn).mean(axis=-1, keepdims=True)
        dx = (t - xn * c) / rms
        return dx.astype(self.dtype), dw.astype(self.dtype)

    # ---------------- backward de SwiGLU ----------------
    def _swiglu_bwd(self, pref: str, dout: np.ndarray, x: np.ndarray, C: dict,
                    acum: Callable) -> np.ndarray:
        """
        Cadena completa de un SwiGLU, hacia atrás. Con x:(N,d), g,u,a:(N,h),
        wd:(h,d) y out:(N,d):

            out = a @ wd          ->  dwd = a.T @ dout   (h,d)
                                      dA  = dout @ wd.T  (N,h)
            a   = silu(g) * u     ->  du  = dA * silu(g)
                                      dg  = dA * u * silu'(g)
            silu'(g) = σ(g)·(1 + g·(1−σ(g)))
            g = x @ wg            ->  dwg = x.T @ dg ;  dx = dg @ wg.T
            u = x @ wu            ->  dwu = x.T @ du ;  dx += du @ wu.T

        `dout` es dL/d(out), la derivada respecto a la SALIDA del bloque. No
        confundir con dL/d(x): son tensors distintos en puntos distintos de la
        cadena, y mezclarlos da el fallo silencioso más típico de un backward
        escrito a mano.
        """
        a = C[(pref, "a")]                                     # (N, ffn_hidden)
        acum(f"{pref}.wd", a.T @ dout)                         # (ffn_hidden, d_model)
        dA = dout @ self.p[f"{pref}.wd"].T                     # (N, ffn_hidden)
        g, u, sg = C[(pref, "g")], C[(pref, "u")], C[(pref, "sg")]
        du = dA * sg
        s = sigmoid(g)
        dg = (dA * u) * s * (1.0 + g * (1.0 - s))
        acum(f"{pref}.wg", x.T @ dg)
        acum(f"{pref}.wu", x.T @ du)
        return dg @ self.p[f"{pref}.wg"].T + du @ self.p[f"{pref}.wu"].T

    def _ffn_bwd(self, pre: str, dout: np.ndarray, C: dict) -> np.ndarray:
        """
        dout es dL/d(salida del FFN), que coincide con dL/d(xn2) propagado por
        la norma. Devuelve dL/d(xn2), que es lo que se suma al residual.

        Formas: xn2:(N,d)  g,u,sg,a:(N,h)  wd:(h,d)  -> todo en 2D.
        """
        xn2 = C[f"{pre}.xn2"].reshape(-1, self.cfg.d_model)     # (N, d)
        dout2 = dout.reshape(-1, self.cfg.d_model)              # (N, d)
        a = C[f"{pre}.a"]                                       # (N, h)

        self.g[f"{pre}.wd"] += a.T @ dout2                      # (h, d)
        dA = dout2 @ self.p[f"{pre}.wd"].T                      # (N, h)

        g, u, sg = C[f"{pre}.g"], C[f"{pre}.u"], C[f"{pre}.sg"]
        du = dA * sg
        s_ = sigmoid(g)
        dg = (dA * u) * s_ * (1.0 + g * (1.0 - s_))

        self.g[f"{pre}.wg"] += xn2.T @ dg                       # (d, h)
        self.g[f"{pre}.wu"] += xn2.T @ du
        return (dg @ self.p[f"{pre}.wg"].T + du @ self.p[f"{pre}.wu"].T).reshape(dout.shape)

    def _moe_bwd(self, pre: str, dY: np.ndarray, C: dict, _2d) -> np.ndarray:
        cfg = self.cfg
        k, d = cfg.top_k, cfg.d_model
        dflat = dY.reshape(-1, d)
        flat = C[f"{pre}.flat"]                       # (N,d), ya aplanado en fwd
        idx = C[f"{pre}.idx"]
        g_norm = C[f"{pre}.g_norm"]
        g_sel = C[f"{pre}.g_sel"]
        sig = C[f"{pre}.sig"]
        suma = C[f"{pre}.g_sum"]
        N = flat.shape[0]

        def acum(n, grad):
            self.g[n] += grad

        dx = np.zeros_like(flat)

        # ---- expertos compartidos ----
        for s in range(cfg.shared_experts):
            sp = f"{pre}.s{s}"
            dx += self._swiglu_bwd(sp, dflat, flat, C, acum)

        # ---- expertos ruteados ----
        dg_norm = np.zeros_like(g_norm)
        for j in range(k):
            a_j = C[f"{pre}.exp_a"][j]
            dg_norm[:, j] = (dflat * a_j).sum(axis=1)
            dj = dflat * g_norm[:, j:j + 1]

            e = idx[:, j]
            for eid in np.unique(e):
                xs, g_, u_, sg_, a_, sel = C[(f"{pre}.e", int(eid), j)]
                djs = dj[sel]
                p = f"{pre}.e{eid}"
                acum(f"{p}.wd", a_.T @ djs)
                dA_ = djs @ self.p[f"{p}.wd"].T
                du_ = dA_ * sg_
                s_ = sigmoid(g_)
                dg_ = (dA_ * u_) * s_ * (1.0 + g_ * (1.0 - s_))
                acum(f"{p}.wg", xs.T @ dg_)
                acum(f"{p}.wu", xs.T @ du_)
                dx[sel] += dg_ @ self.p[f"{p}.wg"].T + du_ @ self.p[f"{p}.wu"].T

        # ---- gate normalizado -> sigmoid -> router ----
        # g_norm_j = g_sel_j / Σ ;  dg_sel_j = (dg_norm_j·Σ − g_sel_j·Σ dg_norm) / Σ²
        S = suma[:, 0]
        total = dg_norm.sum(axis=1, keepdims=True)[:, 0]
        dg_sel = (dg_norm * S[:, None] - g_sel * total[:, None]) / (S[:, None] ** 2)

        dsig = np.zeros_like(sig)
        np.add.at(dsig, (np.arange(N)[:, None], idx), dg_sel)
        drl = dsig * sig * (1.0 - sig)

        acum(f"{pre}.router", flat.T @ drl)
        dx += drl @ self.p[f"{pre}.router"].T
        return dx.reshape(dY.shape)

    # ================================================================== #
    def _balance_update(self, C: dict) -> None:
        """
        Balanceo sin loss auxiliar (DeepSeek-V3).

        Por cada experto comparamos su frecuencia de uso con el objetivo
        (1/n_experts). Si se usó de más, le SUBIMOS el bias, y como en el
        forward hacemos `rl - bias`, subirle el bias le baja la puntuación la
        próxima vez. Y al revés. Es un controlador proporcional que vive fuera
        del gradiente: no distorsiona la pérdida en absoluto.
        """
        tgt = self.cfg.balance_target
        lr = self.cfg.balance_bias_lr
        for l in range(self.cfg.n_layers):
            if not self.es_moe[l]:
                continue
            uso = C.get(f"l{l}.uso")
            if uso is None:
                continue
            b = self.p[f"l{l}.balance_bias"]
            b += lr * (uso - tgt)
            np.clip(b, -0.5, 0.5, out=b)

    def uso_expertos(self) -> dict[str, np.ndarray]:
        """Frecuencia de uso por experto de la última forward. Para vigilar el MoE."""
        return {f"l{l}": self._C.get(f"l{l}.uso") for l in range(self.cfg.n_layers)
                if self._C.get(f"l{l}.uso") is not None}

    # ================================================================== #
    def save(self, ruta: str | Path) -> Path:
        ruta = Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(ruta.with_suffix(".npz"), **self.p)
        ruta.with_suffix(".cfg.json").write_text(
            json.dumps(self.cfg.to_dict(), indent=2), encoding="utf-8")
        return ruta

    @classmethod
    def load(cls, ruta: str | Path, dtype=np.float32) -> "Modelo":
        ruta = Path(ruta)
        cfg = ModelConfig.from_dict(
            json.loads(ruta.with_suffix(".cfg.json").read_text(encoding="utf-8")))
        m = cls(cfg, dtype=dtype)
        datos = np.load(ruta.with_suffix(".npz"))
        for k in list(m.p):
            if k in datos:
                m.p[k] = datos[k].astype(np.float32 if dtype == np.float32 else dtype)
        return m

    # ------------------------------------------------------------------ #
    # Generación                                                          #
    # ------------------------------------------------------------------ #
    def generar(self, ids_prompt: Iterable[int], eos: int,
                max_nuevos: int = 128, temperatura: float = 0.8,
                top_k: int = 10, semilla: Optional[int] = None,
                suprimir: Iterable[int] = ()) -> list[int]:
        """
        Muestreo autoregresivo: la parte en la que el modelo HABLA.

        Dado un prompt, genera un token cada vez hasta que sale `eos` o se
        alcanza `max_nuevos`. Devuelve SOLO los tokens nuevos, sin el prompt.

        Sin caché KV, a propósito: con este tamaño (6 capas, ctx 256) recalcular
        el forward completo en cada paso cuesta decenas de milisegundos y nos
        ahorra dos caminos de código que puedan divergir entre entrenamiento y
        generación — el clásico bug de "en los tests funciona y en producción
        no". La caché es la primera optimización cuando el modelo crezca; es
        un asunto de velocidad, no de corrección.

        Reglas, en orden de importancia:
          1. Lo que está en `suprimir` no se muestrea NUNCA. Así se impide que
             el modelo suelte <user> o <ctx> a mitad de una respuesta: los
             tokens de formato son andamio, no palabras. Si se suprime también
             `eos`, la generación para solo al agotar `max_nuevos` o el
             contexto; si se suprime absolutamente todo, el bucle emite `eos`
             y para en vez de generar basura.
          2. temperatura <= 0 -> voraz (siempre el token más probable). Dos
             llamadas con el mismo prompt dan el mismo texto: es el modo que
             usan los tests y el que sirve para depurar.
          3. top_k recorta la distribución antes de muestrear. En un modelo
             pequeño la cola de la distribución es ruido puro; dejarla viva es
             invitarlo a decir basura con mucha confianza.
        """
        rng = np.random.default_rng(semilla)
        seq = [int(i) for i in ids_prompt]
        assert seq, "el prompt está vacío: no hay nada que continuar"
        assert 0 <= eos < self.cfg.vocab_size, f"eos={eos} fuera de vocabulario"

        prohibidos = sorted({int(s) for s in suprimir})
        nuevos: list[int] = []
        limite = self.cfg.max_seq_len    # forward revienta si nos pasamos

        while len(nuevos) < max_nuevos and len(seq) < limite:
            _, logits = self.forward(np.array([seq], dtype=np.int64))
            # float64 para el softmax: con float32, exp() de logits grandes se
            # come la precisión justo donde más importa, en las colas.
            lg = np.asarray(logits[0, -1], dtype=np.float64)
            if prohibidos:
                lg[prohibidos] = -np.inf
            if np.all(np.isneginf(lg)):
                # Todo suprimido: la única salida sensata es parar. Emitir el
                # argmax de una fila de -inf sería generar justo lo prohibido.
                nuevos.append(eos)
                break

            if temperatura <= 0:
                tok = int(np.argmax(lg))
            else:
                p = np.exp((lg - lg.max()) / temperatura)
                if top_k and 0 < top_k < len(p):
                    umbral = np.partition(p, -top_k)[-top_k]
                    p = np.where(p >= umbral, p, 0.0)
                total = p.sum()
                if not np.isfinite(total) or total <= 0:
                    # Todo suprimido o desbordado: voraz como red de seguridad.
                    tok = int(np.argmax(lg))
                else:
                    tok = int(rng.choice(len(p), p=p / total))

            nuevos.append(tok)
            seq.append(tok)
            if tok == eos:
                break
        return nuevos


# ===========================================================================
# Optimizador
# ===========================================================================
class AdamW:
    """
    Adam con desacoplamiento del weight decay, que es lo que usan todos los
    LLM. El decay se aplica directamente a los pesos, NO dentro del momento,
    que es la diferencia con Adam clásico y por lo que regulariza de verdad.

    Los bias de balanceo del MoE se excluyen: tienen su propia regla.
    Tampoco aplicamos decay a las normas ni a los embeddings (convención
    estándar: regularizar una norma no tiene sentido).
    """

    def __init__(self, modelo: Modelo, lr: float = 3e-4, betas=(0.9, 0.95),
                 eps: float = 1e-8, weight_decay: float = 0.1):
        self.m = modelo
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, *betas, eps, weight_decay
        self.t = 0
        self.nombres = modelo.params_entrenables
        self.sin_decay = {k for k in self.nombres
                          if k.endswith(".w") and self.m.p[k].ndim == 1}
        self.sin_decay |= {"emb.w"}
        self.mom1 = {k: np.zeros_like(modelo.p[k]) for k in self.nombres}
        self.mom2 = {k: np.zeros_like(modelo.p[k]) for k in self.nombres}

    def step(self, grad_clip: float = 1.0) -> float:
        self.t += 1
        # recorte global por norma: si un lote da un gradiente enorme, se escala
        # entero. Evita los picos de pérdida que matan entrenamientos.
        norma = math.sqrt(sum(float((self.m.g[k] ** 2).sum()) for k in self.nombres))
        escala = min(1.0, grad_clip / (norma + 1e-9))

        c1 = 1.0 - self.b1 ** self.t          # correcciones de sesgo: al
        c2 = 1.0 - self.b2 ** self.t          # principio los momentos están
        for k in self.nombres:                # sesgados hacia 0 y hay que
            g = self.m.g[k] * escala          # compensarlo
            self.mom1[k] = self.b1 * self.mom1[k] + (1 - self.b1) * g
            self.mom2[k] = self.b2 * self.mom2[k] + (1 - self.b2) * (g * g)
            m_hat = self.mom1[k] / c1
            v_hat = self.mom2[k] / c2
            paso = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            if self.wd and k not in self.sin_decay:
                paso += self.lr * self.wd * self.m.p[k]
            self.m.p[k] -= paso
        return norma

    def lr_schedule(self, paso: int, total: int, warmup: int = 50,
                    lr_min_ratio: float = 0.1) -> float:
        """Warmup lineal + decaimiento coseno. Es el schedule estándar."""
        if paso < warmup:
            self.lr = self.lr_base * (paso + 1) / warmup
        else:
            prog = (paso - warmup) / max(1, total - warmup)
            cos = 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
            self.lr = self.lr_base * (lr_min_ratio + (1 - lr_min_ratio) * cos)
        return self.lr

    @property
    def lr_base(self):
        if not hasattr(self, "_lr_base"):
            self._lr_base = self.lr
        return self._lr_base
