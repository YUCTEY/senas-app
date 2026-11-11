// Inferencia en el navegador (equivale a modelo_numpy.py): conv 3x3 'same'+ReLU, BN, maxpool 2x2, densas.
const b64 = (s) => { const t = atob(s), u = new Uint8Array(t.length); for (let i = 0; i < t.length; i++) u[i] = t.charCodeAt(i); return new Float32Array(u.buffer) }

export async function cargarModelo(url) {
  const j = await (await fetch(url)).json()
  return j.capas.map((c) => {
    const o = { t: c.t, ks: c.ks, eps: c.eps, act: c.act }
    for (const k of ['k', 'b', 'g', 'be', 'mu', 'var', 'w']) if (c[k]) o[k] = b64(c[k])
    return o
  })
}

// x: Float32Array de 784 valores (28x28, 0-1). Devuelve las 24 probabilidades.
export function predecir(capas, input) {
  let x = input, h = 28, w = 28, c = 1
  for (const L of capas) {
    if (L.t === 'conv') {
      const [kh, kw, ci, co] = L.ks, y = new Float32Array(h * w * co)
      for (let i = 0; i < h; i++) for (let j = 0; j < w; j++) for (let o = 0; o < co; o++) {
        let s = L.b[o]
        for (let a = 0; a < kh; a++) { const ii = i + a - 1; if (ii < 0 || ii >= h) continue
          for (let b = 0; b < kw; b++) { const jj = j + b - 1; if (jj < 0 || jj >= w) continue
            const xo = (ii * w + jj) * ci, ko = (a * kw + b) * ci * co + o
            for (let q = 0; q < ci; q++) s += x[xo + q] * L.k[ko + q * co] } }
        y[(i * w + j) * co + o] = s > 0 ? s : 0
      }
      x = y; c = co
    } else if (L.t === 'bn') {
      const y = new Float32Array(x.length)
      for (let i = 0; i < x.length; i++) { const ch = i % c; y[i] = L.g[ch] * (x[i] - L.mu[ch]) / Math.sqrt(L.var[ch] + L.eps) + L.be[ch] }
      x = y
    } else if (L.t === 'pool') {
      const h2 = h >> 1, w2 = w >> 1, y = new Float32Array(h2 * w2 * c)
      for (let i = 0; i < h2; i++) for (let j = 0; j < w2; j++) for (let ch = 0; ch < c; ch++) {
        const p = (r, s) => x[((2 * i + r) * w + 2 * j + s) * c + ch]
        y[(i * w2 + j) * c + ch] = Math.max(p(0, 0), p(0, 1), p(1, 0), p(1, 1))
      }
      x = y; h = h2; w = w2
    } else if (L.t === 'dense') {
      const [n, m] = L.ks, y = Float32Array.from(L.b)
      for (let i = 0; i < n; i++) { const xi = x[i]; if (xi === 0) continue; for (let o = 0; o < m; o++) y[o] += xi * L.w[i * m + o] }
      if (L.act === 'relu') for (let o = 0; o < m; o++) y[o] = Math.max(0, y[o])
      else { const mx = Math.max(...y); let s = 0; for (let o = 0; o < m; o++) { y[o] = Math.exp(y[o] - mx); s += y[o] } for (let o = 0; o < m; o++) y[o] /= s }
      x = y
    } // 'flatten': ya está en orden alto-ancho-canal
  }
  return x
}
