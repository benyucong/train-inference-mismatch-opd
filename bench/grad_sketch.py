"""CountSketch of parameter-space gradients, for the prefix-shift measurement (prefix_shift.py).

A sketch S maps a vector x over all of the student's parameters (about 7.6e9 numbers) to k = 2^23 numbers:

    (S x)[b] = sum over the elements i with h(i) = b of s(i) x_i,      h(i) uniform on [0, k), s(i) = +1 or -1.

For two vectors <S x, S y> is an unbiased estimate of <x, y>; its standard deviation is at most sqrt(2 / k) ||x|| ||y||
(4.9e-4 ||x|| ||y|| at k = 2^23). S is linear, so the sketch of any linear combination of fields is the same combination
of their sketches: every norm, inner product and cosine the reader needs comes from the Gram matrix of the sketches.

h and s come from one random integer in [0, 2^(log2k + 1)) per element: the low log2k bits are the bucket, the next bit
is the sign. The integers come from a torch.Generator on the tensor's own device, seeded per (sketch seed, parameter
name, chunk index), so one seed gives the same h and s for the same parameter in every call. CPU and CUDA generators
give different streams: a sketch can only be compared with sketches made on the same kind of device.

SketchAcc has probe0_lib.FieldAcc's interface (add_body, add_lm), so probe0_lib.pullback_trace drives it unchanged. The
lm_head gradient of one sequence is summed in fp32 on the GPU (as FieldAcc does) and sketched once per sequence.
"""
import hashlib

import torch

SKETCH_LOG2 = 23            # k = 2^23
SKETCH_SEED = 20261018      # the hash functions' seed (any fixed number; the reader records it)
CHUNK = 1 << 24             # elements per generator call
LM_NAME = "lm_head.weight"  # FieldAcc's name for the output layer


def stream_seed(seed, name, chunk):
    """Stable 63-bit seed for (sketch seed, parameter name, chunk index). Python's hash() is salted: never used."""
    return int.from_bytes(hashlib.sha256(f"{seed}:{name}:{chunk}".encode()).digest()[:8], "little") & ((1 << 63) - 1)


class Sketcher:
    def __init__(self, log2k=SKETCH_LOG2, seed=SKETCH_SEED, chunk=CHUNK):
        self.log2k = int(log2k)
        self.k = 1 << self.log2k
        self.seed = int(seed)
        self.chunk = int(chunk)

    def info(self):
        return dict(log2k=self.log2k, k=self.k, seed=self.seed, chunk=self.chunk)

    def codes(self, name, ci, n, device):
        g = torch.Generator(device=device)
        g.manual_seed(stream_seed(self.seed, name, ci))
        return torch.randint(0, 1 << (self.log2k + 1), (n,), generator=g, device=device, dtype=torch.int32)

    def add(self, out, name, x, alpha=1.0):
        """out += alpha * S(x) for the parameter `name` (out: [k] float tensor on x's device)."""
        flat = x.detach().reshape(-1)
        n = flat.numel()
        for ci, c0 in enumerate(range(0, n, self.chunk)):
            c1 = min(n, c0 + self.chunk)
            r = self.codes(name, ci, c1 - c0, flat.device)
            v = flat[c0:c1].to(out.dtype)
            v = torch.where((r >> self.log2k).bool(), -v, v)
            if alpha != 1.0:
                v = v * alpha
            out.index_add_(0, r & (self.k - 1), v)
            del r, v

    def sketch(self, named, device=None, dtype=torch.float32):
        """S(x) for x given as name -> tensor (e.g. FieldAcc.tensors()); computed on `device` (default: each
        tensor's own device, which must then be the same for all)."""
        out = None
        for name, t in named.items():
            tt = t if device is None else t.to(device)
            if out is None:
                out = torch.zeros(self.k, dtype=dtype, device=tt.device)
            self.add(out, name, tt)
        return out


class SketchAcc:
    """FieldAcc's interface for probe0_lib.pullback_trace. body_names: the body parameters in param_list order."""

    def __init__(self, sk, body_names, lm_shape, device, lm_name=LM_NAME):
        self.sk = sk
        self.names = list(body_names)
        self.lm_name = lm_name
        self.vec = torch.zeros(sk.k, dtype=torch.float32, device=device)
        self.lm = torch.zeros(tuple(lm_shape), dtype=torch.float32, device=device)
        self.lm_dirty = False

    def add_body(self, grads):
        for name, g in zip(self.names, grads):
            if g is not None:
                self.sk.add(self.vec, name, g)

    def add_lm(self, g):
        self.lm.add_(g.to(torch.float32))
        self.lm_dirty = True

    def flush(self):
        if self.lm_dirty:
            self.sk.add(self.vec, self.lm_name, self.lm)
            self.lm.zero_()
            self.lm_dirty = False

    def take(self):
        """This sequence's (or these sequences') sketch; the accumulator is zeroed."""
        self.flush()
        out = self.vec.clone()
        self.vec.zero_()
        return out


class TeeAcc:
    """Feeds the same gradients to several accumulators (the smoke's exact FieldAcc beside the sketch)."""

    def __init__(self, *accs):
        self.accs = accs

    def add_body(self, grads):
        for a in self.accs:
            a.add_body(grads)

    def add_lm(self, g):
        for a in self.accs:
            a.add_lm(g)


def gram(X, device, col_chunk=1 << 18):
    """X: [n, k] float32 (host) -> the [n, n] float64 Gram matrix X X^T, accumulated over column blocks on `device`
    in float64. Returns a CPU float64 tensor."""
    n, k = X.shape
    G = torch.zeros(n, n, dtype=torch.float64, device=device)
    for c0 in range(0, k, col_chunk):
        Y = X[:, c0:min(k, c0 + col_chunk)].to(device=device, dtype=torch.float64)
        G.addmm_(Y, Y.T)
        del Y
    return G.cpu()
