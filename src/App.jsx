import { useEffect, useRef, useState } from 'react'
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision'
import { cargarModelo, predecir } from './modelo'

const BASE = import.meta.env.BASE_URL
const url = (p) => new URL(BASE + p, window.location.href).href
const LETRAS = 'ABCDEFGHIKLMNOPQRSTUVWXY'.split('') // 24 clases (sin J ni Z)
const MODELOS = { cnn: ['CNN mejorada', 'modelos/cnn.json'], mlp: ['MLP baseline', 'modelos/mlp.json'] }
const CONFIRMAR = 3, INTERVALO = 150, VENTANA = 4, T = 112, S = 320 // T = 28*4 (promedio por bloques = filtro BOX)
let lienzo

// Cuadrado alrededor de la mano a partir de sus 21 puntos (con margen).
function cuadro(pts, vw, vh) {
  const xs = pts.map((p) => p.x * vw), ys = pts.map((p) => p.y * vh)
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys)
  const lado = Math.max(x1 - x0, y1 - y0) * 1.35
  return { x: (x0 + x1) / 2 - lado / 2, y: (y0 + y1) / 2 - lado / 2, lado }
}

// Dibuja los puntos de la mano y el cuadro de recorte sobre el video (que se muestra recortado a cuadrado).
function dibujar(cv, pts, r, vw, vh) {
  const g = cv.getContext('2d'); g.clearRect(0, 0, S, S)
  if (!pts) return
  const e = S / Math.min(vw, vh), ox = (vw * e - S) / 2, oy = (vh * e - S) / 2
  const P = (x, y) => [x * e - ox, y * e - oy]
  g.strokeStyle = '#22c55e'; g.lineWidth = 3; g.strokeRect(...P(r.x, r.y), r.lado * e, r.lado * e)
  g.strokeStyle = '#fff'; g.lineWidth = 2
  for (const { start, end } of HandLandmarker.HAND_CONNECTIONS) {
    g.beginPath(); g.moveTo(...P(pts[start].x * vw, pts[start].y * vh)); g.lineTo(...P(pts[end].x * vw, pts[end].y * vh)); g.stroke()
  }
  g.fillStyle = '#3b82f6'
  for (const p of pts) { const [x, y] = P(p.x * vw, p.y * vh); g.beginPath(); g.arc(x, y, 4, 0, 7); g.fill() }
}

// Recorta la mano, pasa a gris, (autocontraste opcional) y reduce a 28x28 en [0,1].
function preparar(video, r, izq, contraste) {
  lienzo ||= Object.assign(document.createElement('canvas'), { width: T, height: T })
  const ctx = lienzo.getContext('2d', { willReadFrequently: true })
  ctx.fillStyle = '#777'; ctx.fillRect(0, 0, T, T)
  ctx.save(); if (izq) { ctx.translate(T, 0); ctx.scale(-1, 1) }
  ctx.drawImage(video, r.x, r.y, r.lado, r.lado, 0, 0, T, T); ctx.restore()
  const d = ctx.getImageData(0, 0, T, T).data, g = new Float32Array(T * T)
  for (let i = 0; i < g.length; i++) g[i] = 0.299 * d[4 * i] + 0.587 * d[4 * i + 1] + 0.114 * d[4 * i + 2]
  let lo = 0, hi = 255
  if (contraste) {
    const h = new Array(256).fill(0), corte = g.length * 0.01; g.forEach((v) => h[Math.round(v)]++)
    let s = 0; while (lo < 255 && s + h[lo] <= corte) s += h[lo++]
    s = 0; while (hi > 0 && s + h[hi] <= corte) s += h[hi--]
  }
  const esc = hi > lo ? 255 / (hi - lo) : 1, out = new Float32Array(784), k = T / 28
  for (let y = 0; y < 28; y++) for (let x = 0; x < 28; x++) {
    let s = 0
    for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) s += Math.min(255, Math.max(0, (g[(y * k + a) * T + x * k + b] - lo) * esc))
    out[y * 28 + x] = s / (k * k) / 255
  }
  return out
}

export default function App() {
  const video = useRef(), over = useRef(), vista = useRef(), modelo = useRef(null), lm = useRef(null)
  const est = useRef({ letra: null, n: 0, aceptada: null, buf: [] })
  const [mk, setMk] = useState('cnn'), [umbral, setUmbral] = useState(80), [contraste, setContraste] = useState(false)
  const [izq, setIzq] = useState(false), [cam, setCam] = useState('user'), [activa, setActiva] = useState(false)
  const [texto, setTexto] = useState(''), [res, setRes] = useState(null), [err, setErr] = useState('')
  const cfg = useRef({}); cfg.current = { umbral, contraste, izq }

  useEffect(() => { // modelo de clasificación elegido
    modelo.current = null
    cargarModelo(url(MODELOS[mk][1])).then((m) => (modelo.current = m)).catch(() => setErr('No se pudo cargar el modelo.'))
  }, [mk])

  useEffect(() => { // detector de manos (MediaPipe), archivos propios en public/
    (async () => {
      try {
        const fs = await FilesetResolver.forVisionTasks(url('mediapipe/wasm'))
        const op = (d) => ({ baseOptions: { modelAssetPath: url('modelos/hand_landmarker.task'), delegate: d }, runningMode: 'VIDEO', numHands: 1 })
        try { lm.current = await HandLandmarker.createFromOptions(fs, op('GPU')) } catch { lm.current = await HandLandmarker.createFromOptions(fs, op('CPU')) }
      } catch { setErr('No se pudo cargar el detector de manos: revisa que existan public/modelos/hand_landmarker.task y public/mediapipe/wasm.') }
    })()
  }, [])

  async function iniciar() {
    setErr('')
    if (!navigator.mediaDevices?.getUserMedia) return setErr('La cámara requiere HTTPS (o localhost).')
    try {
      video.current.srcObject?.getTracks().forEach((t) => t.stop())
      video.current.srcObject = await navigator.mediaDevices.getUserMedia({ video: { facingMode: cam, width: { ideal: 640 }, height: { ideal: 480 } }, audio: false })
      await video.current.play(); setActiva(true)
    } catch { setErr('No se pudo abrir la cámara. Revisa el permiso del navegador.') }
  }
  useEffect(() => { if (activa) iniciar() }, [cam])

  useEffect(() => {
    if (!activa) return
    const id = setInterval(() => {
      const v = video.current, m = modelo.current, d = lm.current, e = est.current
      if (!m || !d || !v.videoWidth) return
      const vw = v.videoWidth, vh = v.videoHeight, pts = d.detectForVideo(v, performance.now()).landmarks[0]
      if (!pts) { dibujar(over.current, null); Object.assign(e, { letra: null, n: 0, aceptada: null, buf: [] }); return setRes({ sinMano: true }) }
      const caja = cuadro(pts, vw, vh); dibujar(over.current, pts, caja, vw, vh)
      const x = preparar(v, caja, cfg.current.izq, cfg.current.contraste)
      const cv = vista.current.getContext('2d'), im = cv.createImageData(28, 28)
      x.forEach((val, i) => { im.data.set([val * 255, val * 255, val * 255, 255], i * 4) }); cv.putImageData(im, 0, 0)
      e.buf.push(predecir(m, x)); if (e.buf.length > VENTANA) e.buf.shift()
      const p = e.buf[0].map((_, i) => e.buf.reduce((s, q) => s + q[i], 0) / e.buf.length) // promedio de los últimos cuadros
      const top = [...p.keys()].sort((a, b) => p[b] - p[a]).slice(0, 3)
      const letra = LETRAS[top[0]], conf = p[top[0]], ok = conf >= cfg.current.umbral / 100
      if (ok) {
        e.n = letra === e.letra ? e.n + 1 : 1; e.letra = letra
        if (e.n >= CONFIRMAR && letra !== e.aceptada) { e.aceptada = letra; setTexto((t) => t + letra) }
      } else { e.n = 0; e.letra = null; e.aceptada = null }
      setRes({ letra, conf, ok, n: Math.min(e.n, CONFIRMAR), top: top.map((i) => [LETRAS[i], p[i]]) })
    }, INTERVALO)
    return () => clearInterval(id)
  }, [activa])

  const estado = !res ? 'Pulsa “Iniciar cámara” y muestra tu mano'
    : res.sinMano ? 'Sin mano detectada'
    : res.ok ? `Confianza ${Math.round(res.conf * 100)} % · estable ${res.n}/${CONFIRMAR}`
    : `Baja confianza (${Math.round(res.conf * 100)} %). Mantén la seña quieta.`
  return (
    <main>
      <h1>🤟 Reconocimiento de señas</h1>
      <div className="visor" style={{ transform: cam === 'user' ? 'scaleX(-1)' : 'none' }}>
        <video ref={video} playsInline muted />
        <canvas ref={over} className="over" width={S} height={S} />
      </div>
      <div className="fila">
        <button onClick={iniciar}>{activa ? 'Reiniciar cámara' : 'Iniciar cámara'}</button>
        <button className="sec" onClick={() => setCam(cam === 'user' ? 'environment' : 'user')}>Cambiar cámara</button>
      </div>
      {err && <p className="err">{err}</p>}
      <section className="panel">
        <div className="letra" style={{ opacity: res?.ok ? 1 : 0.35 }}>{res?.letra ?? '–'}</div>
        <div className="barra"><div style={{ width: `${(res?.conf ?? 0) * 100}%`, background: res?.ok ? 'var(--ok)' : 'var(--mal)' }} /></div>
        <p className="estado">{estado}</p>
        <p className="top">{res?.top?.map(([l, c]) => `${l} ${Math.round(c * 100)}%`).join('  ·  ')}</p>
      </section>
      <section className="panel">
        <div className="texto">{texto || <span className="vacio">El texto aparecerá aquí…</span>}</div>
        <div className="fila">
          <button className="sec" onClick={() => setTexto((t) => t.slice(0, -1))}>Borrar última</button>
          <button className="sec" onClick={() => setTexto((t) => t + ' ')}>Espacio</button>
          <button className="sec" onClick={() => setTexto('')}>Limpiar</button>
        </div>
      </section>
      <section className="panel ajustes">
        <label>Modelo
          <select value={mk} onChange={(e) => setMk(e.target.value)}>{Object.entries(MODELOS).map(([k, [n]]) => <option key={k} value={k}>{n}</option>)}</select>
        </label>
        <label>Confianza mínima: <b>{umbral} %</b><input type="range" min="40" max="95" step="5" value={umbral} onChange={(e) => setUmbral(+e.target.value)} /></label>
        <label className="chk"><input type="checkbox" checked={contraste} onChange={(e) => setContraste(e.target.checked)} /> Mejorar contraste</label>
        <label className="chk"><input type="checkbox" checked={izq} onChange={(e) => setIzq(e.target.checked)} /> Mano izquierda (voltear)</label>
        <div className="mini"><canvas ref={vista} width="28" height="28" /><span>Lo que ve el modelo (28×28, gris): la mano recortada automáticamente. Todo se procesa en tu dispositivo: no se envía ni se guarda ninguna imagen.</span></div>
      </section>
    </main>
  )
}