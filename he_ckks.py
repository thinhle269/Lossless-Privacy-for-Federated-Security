"""CKKS homomorphic secure aggregation for federated averaging.

A client encrypts its flattened model-update vector; the server computes the
weighted sum **on ciphertexts** (it never sees a plaintext update); the result is
decrypted only as the aggregate. The parameter vector is packed into fixed-size
chunks of `slot_count = poly_modulus_degree / 2` slots.

Backends
--------
tenseal  : real CKKS (measured overhead).  Preferred and auto-detected.
emulated : portable NumPy fallback that reproduces CKKS approximation error and
           reports ciphertext size from an analytical model (overhead = ESTIMATE).
           Used only if TenSEAL is unavailable, and clearly flagged in outputs.
"""
import time
import numpy as np

try:
    import tenseal as ts
    HAS_TENSEAL = True
except Exception:
    HAS_TENSEAL = False


class CKKSAggregator:
    def __init__(self, poly_modulus_degree=8192, coeff_mod_bit_sizes=(60, 40, 40, 60),
                 scale_bits=40, backend="auto"):
        self.poly = int(poly_modulus_degree)
        self.coeff = list(coeff_mod_bit_sizes)
        self.scale_bits = int(scale_bits)
        self.slots = self.poly // 2
        if backend == "auto":
            backend = "tenseal" if HAS_TENSEAL else "emulated"
        if backend == "tenseal" and not HAS_TENSEAL:
            backend = "emulated"
        self.backend = backend

        if self.backend == "tenseal":
            ctx = ts.context(ts.SCHEME_TYPE.CKKS,
                             poly_modulus_degree=self.poly,
                             coeff_mod_bit_sizes=self.coeff)
            ctx.global_scale = 2 ** self.scale_bits
            ctx.generate_galois_keys()
            self.ctx = ctx
        else:
            self.ctx = None

    # --- helpers -----------------------------------------------------------
    def _split(self, vec):
        return [vec[i:i + self.slots] for i in range(0, len(vec), self.slots)]

    def _emulated_chunk_bytes(self):
        # a fresh CKKS ciphertext ~ 2 polynomials of `poly` coeffs at the top
        # modulus bit-width; this matches TenSEAL serialized sizes within ~2x.
        top_bits = self.coeff[0]
        return int(2 * self.poly * top_bits / 8)

    # --- client: encrypt ---------------------------------------------------
    def encrypt(self, vec):
        """Return (cipher_obj, encrypt_seconds, uploaded_bytes)."""
        chunks = self._split(np.asarray(vec, dtype=np.float64))
        t0 = time.perf_counter()
        if self.backend == "tenseal":
            enc = [ts.ckks_vector(self.ctx, c.tolist()) for c in chunks]
            dt = time.perf_counter() - t0
            nbytes = sum(len(e.serialize()) for e in enc)
        else:
            enc = [c.copy() for c in chunks]
            dt = time.perf_counter() - t0
            nbytes = len(chunks) * self._emulated_chunk_bytes()
        return enc, dt, nbytes

    # --- server: aggregate on ciphertexts ----------------------------------
    def aggregate(self, enc_list, weights):
        """Weighted sum across clients, chunk by chunk. Returns (agg, seconds)."""
        weights = [float(w) for w in weights]
        n_chunks = len(enc_list[0])
        t0 = time.perf_counter()
        out = []
        for ci in range(n_chunks):
            acc = enc_list[0][ci] * weights[0]
            for k in range(1, len(enc_list)):
                acc = acc + enc_list[k][ci] * weights[k]
            out.append(acc)
        dt = time.perf_counter() - t0
        return out, dt

    # --- client: decrypt aggregate -----------------------------------------
    def decrypt(self, agg, length):
        """Return (plain_vector[:length], decrypt_seconds)."""
        t0 = time.perf_counter()
        if self.backend == "tenseal":
            parts = [np.array(e.decrypt()) for e in agg]
            vec = np.concatenate(parts)[:length]
        else:
            vec = np.concatenate(agg)[:length]
            rel = 2.0 ** (-self.scale_bits)
            vec = vec + np.random.normal(0, rel * (np.std(vec) + 1e-9), size=vec.shape)
            dt = time.perf_counter() - t0
            return vec.astype(np.float64), dt
        dt = time.perf_counter() - t0
        return vec.astype(np.float64), dt

    def info(self):
        return {"he_backend": self.backend, "poly_modulus_degree": self.poly,
                "coeff_mod_bit_sizes": self.coeff, "scale_bits": self.scale_bits,
                "slots": self.slots}
