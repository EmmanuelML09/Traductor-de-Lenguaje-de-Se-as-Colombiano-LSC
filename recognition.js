/* =====================================================================
   recognition.js
   ---------------------------------------------------------------------
   Módulo responsable de:
   1. Cargar el detector de manos (MediaPipe Tasks Vision - HandLandmarker).
   2. Convertir los landmarks de cada fotograma en características:
      - FORMA: posición de los 21 puntos de cada mano relativa a la muñeca
        y a su tamaño (no depende de la distancia a la cámara).
      - POSICIÓN: dónde está la muñeca en la imagen y el tamaño de la mano
        (sirve para calcular la TRAYECTORIA del movimiento).
   3. Detectar solo, sin botones, cuándo EMPIEZA y cuándo TERMINA un
      movimiento (MotionSegmenter), para separar las señas dinámicas.
   4. Clasificar señas ESTÁTICAS (una postura quieta) y DINÁMICAS (un
      movimiento completo) comparándolas con los ejemplos entrenados.

   Cómo se mide la "similitud" (honestidad sobre el porcentaje)
   ---------------------------------------------------------------------
   Para cada seña se calcula, con sus propios ejemplos de entrenamiento,
   cuánto varían entre sí dos ejecuciones de la MISMA seña (distancia al
   ejemplo más cercano de la misma seña, dejando fuera al propio ejemplo).
   De ahí sale una "tolerancia" por seña. La similitud mostrada indica qué
   tan dentro de esa variación natural cae la seña observada:
       100 % = idéntica a un ejemplo entrenado
       ~60 % = en el límite de lo que varían tus propios ejemplos
       0 %   = muy lejos de lo entrenado
   No es una probabilidad estadística ni una medición certificada: es una
   comparación contra lo que los estudiantes entrenaron. Además se rechaza
   el resultado cuando dos señas distintas quedan casi igual de cerca
   (ambigüedad), en lugar de elegir una al azar.
   ===================================================================== */

// MediaPipe se importa de forma dinámica (dentro de initHandLandmarker) para que la
// interfaz funcione aunque la descarga falle; solo la cámara necesita el modelo.
const MEDIAPIPE_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

let handLandmarker = null;
let modelState = "not-loaded"; // "not-loaded" | "loading" | "ready" | "error"
let modelError = null;

export function getModelState() {
  return modelState;
}
export function getModelError() {
  return modelError;
}

/** Descarga e inicializa el modelo de detección de manos. Se debe llamar una sola vez al iniciar la app. */
export async function initHandLandmarker() {
  if (modelState === "ready") return handLandmarker;
  modelState = "loading";
  try {
    const { HandLandmarker, FilesetResolver } = await import(MEDIAPIPE_URL);
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        delegate: "GPU"
      },
      runningMode: "VIDEO",
      numHands: 2 // Hasta 2 manos: necesario para señas que usan ambas manos.
    });
    modelState = "ready";
    return handLandmarker;
  } catch (err) {
    modelState = "error";
    modelError =
      "No se pudo cargar el modelo de detección de manos. Verifica tu conexión a internet la primera vez que usas la aplicación. Detalle: " +
      (err && err.message ? err.message : err);
    throw new Error(modelError);
  }
}

/**
 * Detecta manos en el fotograma actual de un elemento <video>.
 * Devuelve el resultado bruto de MediaPipe o null si el modelo aún no
 * está listo o el video no tiene datos todavía.
 */
export function detectHandsInVideo(videoElement, timestampMs) {
  if (modelState !== "ready" || !handLandmarker) return null;
  if (!videoElement || videoElement.readyState < 2) return null;
  return handLandmarker.detectForVideo(videoElement, timestampMs);
}

/* ---------------------------------------------------------------------
   Características de UN fotograma
   --------------------------------------------------------------------- */

const HAND_DIM = 63;   // 21 puntos x 3 coordenadas
const FRAME_DIM = 126; // 2 manos

const round4 = (x) => Math.round(x * 10000) / 10000;

function distance3D(a, b) {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2);
}

/**
 * Forma de UNA mano: 63 números relativos a la muñeca (posición) y
 * divididos por la distancia muñeca-dedo medio (escala).
 */
export function extractFeatureVector(landmarks) {
  const wrist = landmarks[0];
  const scale = distance3D(wrist, landmarks[9]) || 1e-6;
  const vector = [];
  for (const point of landmarks) {
    vector.push((point.x - wrist.x) / scale);
    vector.push((point.y - wrist.y) / scale);
    vector.push((point.z - wrist.z) / scale);
  }
  return vector;
}

/**
 * Ordena las manos de izquierda a derecha en la imagen. MediaPipe no
 * garantiza un orden estable, y sin un orden fijo la "mano 1" y la
 * "mano 2" se intercambiarían entre fotogramas.
 */
export function sortHands(hands) {
  return hands.slice().sort((a, b) => a[0].x - b[0].x);
}

/**
 * Convierte las manos detectadas en un fotograma de características:
 *  - vector: forma de ambas manos (126 valores; ceros si solo hay una)
 *  - pos:    [x, y, tamaño] de la muñeca de cada mano en la imagen (6 valores)
 *  - count:  número de manos
 */
export function handFrameInfo(sortedHands, t) {
  const vector = new Array(FRAME_DIM).fill(0);
  const pos = [0, 0, 0, 0, 0, 0];
  sortedHands.slice(0, 2).forEach((hand, i) => {
    const shape = extractFeatureVector(hand);
    for (let j = 0; j < HAND_DIM; j++) vector[i * HAND_DIM + j] = round4(shape[j]);
    pos[i * 3] = round4(hand[0].x);
    pos[i * 3 + 1] = round4(hand[0].y);
    pos[i * 3 + 2] = round4(Math.hypot(hand[0].x - hand[9].x, hand[0].y - hand[9].y) || 1e-6);
  });
  return { t, vector, pos, count: Math.min(2, sortedHands.length) };
}

/** Los ejemplos antiguos guardaron 63 valores (una mano): se completan con ceros. */
function normalizeVector(v) {
  if (v.length >= FRAME_DIM) return v;
  return v.concat(new Array(FRAME_DIM - v.length).fill(0));
}

function handCountOf(v) {
  return v.slice(HAND_DIM).some((x) => x !== 0) ? 2 : 1;
}

function euclid(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return Math.sqrt(sum);
}

/* ---------------------------------------------------------------------
   Detección automática de INICIO y FINAL de un movimiento
   --------------------------------------------------------------------- */

/**
 * Observa los fotogramas uno por uno y avisa cuándo la(s) mano(s) empiezan
 * a moverse y cuándo se detienen, para entregar el movimiento completo
 * (con un poco de margen al inicio) sin que el usuario presione nada.
 *
 * "Nivel de movimiento" = qué tan rápido se mueve la mano comparado con
 * el umbral de inicio (1 = justo en el umbral). Se mide en tamaños de
 * mano por segundo (así no depende de la distancia a la cámara) y
 * mirando una ventana de unos 160 ms para ignorar el temblor de la
 * detección. Tiene en cuenta tanto el desplazamiento de la mano como el
 * cambio de forma de los dedos.
 *
 * push(frame|null, ahoraMs) devuelve una lista de eventos:
 *   { type: "start" }
 *   { type: "end", reason, frames, durationMs, valid }
 */
export class MotionSegmenter {
  constructor(config = {}) {
    this.cfg = {
      windowMs: 160,        // ventana para medir la velocidad
      posSpeed: 0.9,        // desplazamiento (tamaños de mano / s) que cuenta como movimiento
      shapeSpeed: 4.5,      // cambio de forma de los dedos (unidades / s) que cuenta como movimiento
      startFrames: 2,       // fotogramas seguidos en movimiento para empezar
      stopLevel: 0.55,      // por debajo de este nivel la mano se considera quieta
      stopHoldMs: 420,      // tiempo quieta para dar la seña por terminada
      lostHandMs: 450,      // tiempo sin mano para dar la seña por terminada
      preRollMs: 320,       // margen anterior al momento en que se detectó el inicio
      minDurationMs: 350,   // movimientos más cortos se descartan
      minFrames: 6,
      maxDurationMs: 5000,  // duración máxima de una seña
      sensitivity: 1,       // >1 detecta movimientos más pequeños
      ...config
    };
    this.reset();
  }

  setOptions(options) {
    Object.assign(this.cfg, options);
  }

  reset() {
    this.history = [];
    this.state = "idle"; // "idle" | "moving"
    this.level = 0;
    this.still = false;  // true cuando hay mano y está quieta de forma sostenida
    this.segment = null;
    this.aboveCount = 0;
    this.lastMoveAt = 0;
    this.lastHandAt = 0;
  }

  /** Promedio de hasta 3 fotogramas seguidos que terminan en idx (reduce el temblor de la detección). */
  _smoothed(idx) {
    const last = this.history[idx];
    const group = [last];
    for (let i = idx - 1; i >= 0 && group.length < 3; i--) {
      if (this.history[i].count !== last.count) break;
      group.push(this.history[i]);
    }
    const pos = [0, 0, 0, 0, 0, 0];
    const xy = []; // solo x e y de la forma: la profundidad (z) es la coordenada más ruidosa
    for (let j = 0; j < FRAME_DIM; j++) if (j % 3 !== 2) xy.push(0);
    for (const f of group) {
      for (let j = 0; j < 6; j++) pos[j] += f.pos[j] / group.length;
      let k = 0;
      for (let j = 0; j < FRAME_DIM; j++) if (j % 3 !== 2) xy[k++] += f.vector[j] / group.length;
    }
    return { count: last.count, pos, xy };
  }

  /** Nivel de movimiento del fotograma actual, o null si aún no hay datos suficientes. */
  _level(frame) {
    const { windowMs, posSpeed, shapeSpeed, sensitivity } = this.cfg;
    const last = this.history.length - 1;
    let refIdx = -1;
    for (let i = last - 1; i >= 0; i--) {
      if (this.history[i].t <= frame.t - windowMs) { refIdx = i; break; }
    }
    if (refIdx < 0) return null;

    const cur = this._smoothed(last);
    const ref = this._smoothed(refIdx);
    if (cur.count !== ref.count) return null;

    const seconds = (frame.t - this.history[refIdx].t) / 1000;
    let travel = 0;
    for (let h = 0; h < cur.count; h++) {
      const size = (cur.pos[h * 3 + 2] + ref.pos[h * 3 + 2]) / 2 || 1e-6;
      const d = Math.hypot(cur.pos[h * 3] - ref.pos[h * 3], cur.pos[h * 3 + 1] - ref.pos[h * 3 + 1]) / size;
      if (d > travel) travel = d;
    }
    const shapeChange = euclid(cur.xy, ref.xy);
    return Math.max(
      travel / seconds / (posSpeed / sensitivity),
      shapeChange / seconds / (shapeSpeed / sensitivity)
    );
  }

  push(frame, now) {
    const events = [];

    if (frame) {
      this.lastHandAt = now;
      this.history.push(frame);
      while (this.history.length && this.history[0].t < now - 1500) this.history.shift();

      const level = this._level(frame);
      this.level = level ?? 0;
      this.still = level !== null && level < this.cfg.stopLevel;

      if (this.state === "idle") {
        this.aboveCount = level !== null && level >= 1 ? this.aboveCount + 1 : 0;
        if (this.aboveCount >= this.cfg.startFrames) {
          const from = now - this.cfg.preRollMs;
          this.segment = { frames: this.history.filter((f) => f.t >= from) };
          this.state = "moving";
          this.lastMoveAt = now;
          this.aboveCount = 0;
          events.push({ type: "start" });
        }
      } else {
        this.segment.frames.push(frame);
        if (level !== null && level >= this.cfg.stopLevel) this.lastMoveAt = now;
      }
    } else {
      this.level = 0;
      this.still = false;
      if (this.state === "idle") this.aboveCount = 0;
    }

    if (this.state === "moving") {
      let reason = null;
      if (!frame && now - this.lastHandAt >= this.cfg.lostHandMs) reason = "lost";
      else if (now - this.lastMoveAt >= this.cfg.stopHoldMs) reason = "still";
      else if (now - this.segment.frames[0].t >= this.cfg.maxDurationMs) reason = "max";
      if (reason) events.push(this._finish(reason));
    }
    return events;
  }

  _finish(reason) {
    const all = this.segment.frames;
    // Se descarta el tramo final en que la mano ya estaba quieta (salvo un pequeño margen).
    let kept = all.filter((f) => f.t <= this.lastMoveAt + 150);
    if (kept.length < 2) kept = all;
    const t0 = kept[0].t;
    const frames = kept.map((f) => ({ ...f, t: Math.round(f.t - t0) }));
    const durationMs = frames[frames.length - 1].t;

    this.state = "idle";
    this.segment = null;
    this.aboveCount = 0;
    this.history = []; // evita volver a empezar con datos de la seña que acaba de terminar
    return {
      type: "end",
      reason,
      frames,
      durationMs,
      valid: frames.length >= this.cfg.minFrames && durationMs >= this.cfg.minDurationMs
    };
  }
}

/* ---------------------------------------------------------------------
   Descriptor de un movimiento (forma + trayectoria) y DTW
   --------------------------------------------------------------------- */

const SEQ_LEN = 16;       // cada movimiento se lleva a 16 pasos para poder compararlo
const TRAJ_WEIGHT = 2;    // peso de la trayectoria frente a la forma de la mano

/** Lleva una secuencia de vectores a una longitud fija, interpolando linealmente. */
function resampleSequence(sequence, targetLength = SEQ_LEN) {
  if (sequence.length === 0) return [];
  if (sequence.length === 1) return new Array(targetLength).fill(sequence[0]);
  const result = [];
  for (let i = 0; i < targetLength; i++) {
    const src = (i / (targetLength - 1)) * (sequence.length - 1);
    const lo = Math.floor(src);
    const hi = Math.min(lo + 1, sequence.length - 1);
    const frac = src - lo;
    result.push(sequence[lo].map((v, d) => v + (sequence[hi][d] - v) * frac));
  }
  return result;
}

/**
 * Construye lo que se compara entre dos movimientos:
 *  - shape: la forma de la(s) mano(s) a lo largo del tiempo
 *  - traj:  la trayectoria de cada muñeca respecto a su posición inicial,
 *           medida en tamaños de mano (null en ejemplos antiguos que no la guardaron)
 */
export function segmentDescriptor(frames) {
  const shape = resampleSequence(frames.map((f) => normalizeVector(f.vector)));

  let traj = null;
  if (frames.every((f) => Array.isArray(f.pos))) {
    const first = frames[0].pos;
    const meanSize = [0, 1].map((slot) => {
      const sizes = frames.map((f) => f.pos[slot * 3 + 2]).filter((s) => s > 0);
      return sizes.length ? sizes.reduce((a, b) => a + b, 0) / sizes.length : 0;
    });
    const raw = frames.map((f) => {
      const row = [];
      for (let slot = 0; slot < 2; slot++) {
        const size = meanSize[slot];
        if (!size || f.pos[slot * 3 + 2] === 0 || first[slot * 3 + 2] === 0) {
          row.push(0, 0);
        } else {
          row.push(
            ((f.pos[slot * 3] - first[slot * 3]) / size) * TRAJ_WEIGHT,
            ((f.pos[slot * 3 + 1] - first[slot * 3 + 1]) / size) * TRAJ_WEIGHT
          );
        }
      }
      return row;
    });
    traj = resampleSequence(raw);
  }
  return { shape, traj };
}

function localDistance(a, i, b, j) {
  let sum = 0;
  const sa = a.shape[i], sb = b.shape[j];
  for (let d = 0; d < sa.length; d++) {
    const x = sa[d] - sb[d];
    sum += x * x;
  }
  if (a.traj && b.traj) {
    const ta = a.traj[i], tb = b.traj[j];
    for (let d = 0; d < ta.length; d++) {
      const x = ta[d] - tb[d];
      sum += x * x;
    }
  }
  return Math.sqrt(sum);
}

/**
 * Distancia Dynamic Time Warping entre dos movimientos: alinea ambas
 * secuencias en el tiempo (tolera ir más rápido o más lento) y suma lo
 * que cuesta esa mejor alineación, normalizado por la longitud.
 */
function dtwDistance(a, b) {
  const n = a.shape.length, m = b.shape.length;
  const cost = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(Infinity));
  cost[0][0] = 0;
  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      cost[i][j] = localDistance(a, i - 1, b, j - 1) + Math.min(cost[i - 1][j], cost[i][j - 1], cost[i - 1][j - 1]);
    }
  }
  return cost[n][m] / (n + m);
}

/* ---------------------------------------------------------------------
   Modelos (ejemplos entrenados + tolerancia por seña) y decisión
   --------------------------------------------------------------------- */

const STATIC_TOLERANCE_FLOOR = 0.55;
const DYNAMIC_TOLERANCE_FLOOR = 0.6;

/** Tolerancia de cada seña: cuánto varían entre sí sus propios ejemplos. */
function toleranceByLabel(items, distance, floor) {
  const groups = {};
  items.forEach((it) => (groups[it.label] = groups[it.label] || []).push(it));
  const stats = {};
  for (const [label, list] of Object.entries(groups)) {
    const nearest = [];
    if (list.length >= 2) {
      for (let i = 0; i < list.length; i++) {
        let best = Infinity;
        for (let j = 0; j < list.length; j++) {
          if (i !== j) best = Math.min(best, distance(list[i], list[j]));
        }
        nearest.push(best);
      }
    }
    const mean = nearest.length ? nearest.reduce((a, b) => a + b, 0) / nearest.length : 0;
    const std = nearest.length ? Math.sqrt(nearest.reduce((a, b) => a + (b - mean) ** 2, 0) / nearest.length) : 0;
    stats[label] = { examples: list.length, tolerance: Math.max(mean + 3 * std, mean * 1.5, floor) };
  }
  return stats;
}

/**
 * Elige la seña más cercana y decide si se acepta.
 * Se rechaza si la similitud es baja o si otra seña queda casi igual de cerca.
 */
function decide(candidates, stats, k, minScore) {
  candidates.sort((a, b) => a.dist - b.dist);

  const nearestPerLabel = [];
  const seen = new Set();
  for (const c of candidates) {
    if (!seen.has(c.label)) { seen.add(c.label); nearestPerLabel.push([c.label, c.dist]); }
  }
  const [bestLabel, bestDist] = nearestPerLabel[0];
  const runnerUp = nearestPerLabel[1] || null;

  // Voto ponderado de los k ejemplos más cercanos (los más cercanos pesan más).
  let total = 0, mine = 0;
  for (const c of candidates.slice(0, k)) {
    const w = 1 / (c.dist + 1e-3);
    total += w;
    if (c.label === bestLabel) mine += w;
  }
  const vote = total > 0 ? mine / total : 0;

  const ratio = bestDist / stats[bestLabel].tolerance;
  const typical = Math.max(0, 1 - (ratio / 1.8) ** 2);
  const score = Math.round(100 * typical * vote);

  let accepted = score >= minScore;
  let reason = accepted ? "ok" : "low";
  if (accepted && runnerUp && runnerUp[1] < bestDist * 1.15) {
    accepted = false;
    reason = "ambiguous";
  }
  return { label: bestLabel, score, accepted, reason, runnerUp: runnerUp ? runnerUp[0] : null, distance: bestDist };
}

const NO_MODEL = { label: null, score: 0, accepted: false, reason: "no-model", runnerUp: null, distance: Infinity };

/** Modelo de señas ESTÁTICAS entrenadas de un proyecto. */
export function buildStaticModel(project) {
  const items = [];
  for (const pattern of project.patterns) {
    if (pattern.type !== "static" || !pattern.trained) continue;
    for (const ex of pattern.examples) {
      if (!Array.isArray(ex.landmarks)) continue;
      const vector = normalizeVector(ex.landmarks);
      items.push({ label: pattern.name, vector, hands: handCountOf(vector) });
    }
  }
  return { items, stats: toleranceByLabel(items, (a, b) => euclid(a.vector, b.vector), STATIC_TOLERANCE_FLOOR) };
}

/** Clasifica la postura de un fotograma (vector de 126 valores). */
export function classifyStatic(vector, model, { k = 5, minScore = 65 } = {}) {
  const query = normalizeVector(vector);
  const hands = handCountOf(query);
  // Una seña de una mano no se compara con una postura de dos manos, ni al revés.
  const candidates = model.items
    .filter((it) => it.hands === hands)
    .map((it) => ({ label: it.label, dist: euclid(query, it.vector) }));
  if (candidates.length === 0) return NO_MODEL;
  return decide(candidates, model.stats, k, minScore);
}

/** Modelo de señas DINÁMICAS entrenadas de un proyecto. */
export function buildDynamicModel(project) {
  const items = [];
  for (const pattern of project.patterns) {
    if (pattern.type !== "dynamic" || !pattern.trained) continue;
    for (const ex of pattern.examples) {
      if (Array.isArray(ex.frames) && ex.frames.length >= 2) {
        items.push({ label: pattern.name, descriptor: segmentDescriptor(ex.frames) });
      }
    }
  }
  return {
    items,
    stats: toleranceByLabel(items, (a, b) => dtwDistance(a.descriptor, b.descriptor), DYNAMIC_TOLERANCE_FLOOR)
  };
}

/** Clasifica un movimiento completo (descriptor de segmentDescriptor). */
export function classifyDynamic(descriptor, model, { k = 3, minScore = 60 } = {}) {
  if (model.items.length === 0) return NO_MODEL;
  const candidates = model.items.map((it) => ({ label: it.label, dist: dtwDistance(descriptor, it.descriptor) }));
  return decide(candidates, model.stats, k, minScore);
}
