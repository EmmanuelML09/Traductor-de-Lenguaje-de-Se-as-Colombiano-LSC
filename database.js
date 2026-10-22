/* =====================================================================
   database.js
   ---------------------------------------------------------------------
   Persistencia de los proyectos en el navegador mediante IndexedDB.

   NUEVO MODELO DE DATOS (v2 del prototipo)
   ---------------------------------------------------------------------
   La versión anterior guardaba cada proyecto con una lista "signs"
   (señas estáticas simples). Esta versión la reemplaza por "patterns":
   unidades entrenables de 4 tipos distintos, pensadas para representar
   mejor la naturaleza visual y temporal de la LSC:

     - "static"    seña estática (una configuración de la(s) mano(s))
     - "dynamic"   seña dinámica (una secuencia de movimiento grabada)
     - "sequence"  secuencia de unidades ya entrenadas, en un orden dado
     - "composite" expresión compuesta (igual mecánica que "sequence",
                    pero reservada para expresiones completas que el
                    equipo documentó y verificó como tales)

   MIGRACIÓN AUTOMÁTICA Y SEGURA
   ---------------------------------------------------------------------
   Los proyectos creados con la versión anterior (campo "signs") se
   migran de forma perezosa y transparente la primera vez que se leen:
   cada "seña" antigua se convierte en un patrón de tipo "static",
   conservando TODOS sus ejemplos y landmarks. Nada se borra. El
   proyecto migrado se vuelve a guardar con la nueva forma la próxima
   vez que haya un cambio, por lo que "signs" desaparece del registro
   pero su contenido íntegro pasa a "patterns".
   ===================================================================== */

const DB_NAME = "TraductorLSC_DB";
const DB_VERSION = 1; // No fue necesario subir la versión del esquema: el almacén
                        // de objetos no cambió, solo la FORMA del contenido de cada
                        // proyecto, que se migra en JavaScript (ver migrateProjectShape).
const STORE_NAME = "projects";

let dbInstance = null;

/** Abre (o crea) la base de datos IndexedDB. Devuelve una Promise<IDBDatabase>. */
export function openDatabase() {
  return new Promise((resolve, reject) => {
    if (dbInstance) return resolve(dbInstance);

    if (!("indexedDB" in window)) {
      reject(new Error("Este navegador no soporta IndexedDB. Prueba con Chrome, Edge o Firefox actualizados."));
      return;
    }

    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onupgradeneeded = (event) => {
      const db = event.target.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        const store = db.createObjectStore(STORE_NAME, { keyPath: "id" });
        store.createIndex("updatedAt", "updatedAt", { unique: false });
      }
    };

    request.onsuccess = (event) => {
      dbInstance = event.target.result;
      resolve(dbInstance);
    };

    request.onerror = () => {
      reject(new Error("No se pudo abrir la base de datos local (IndexedDB)."));
    };
  });
}

function generateId(prefix) {
  return `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}

/* ---------------------------------------------------------------------
   Migración perezosa de proyectos antiguos ("signs") al nuevo modelo
   ("patterns"). Se aplica automáticamente en getProject/getAllProjects.
   --------------------------------------------------------------------- */
export function migrateProjectShape(project) {
  if (!project) return project;
  const now = new Date().toISOString();

  if (!Array.isArray(project.patterns)) {
    const legacySigns = Array.isArray(project.signs) ? project.signs : [];
    project.patterns = legacySigns.map((sign) => ({
      id: generateId("pat"),
      name: sign.name,
      type: "static",
      description: "",
      source: "",
      trained: !!sign.trained,
      trainedAt: sign.trainedAt || null,
      createdAt: sign.createdAt || now,
      updatedAt: sign.updatedAt || now,
      sequenceUnits: [],
      examples: (sign.examples || []).map((ex) => ({
        id: ex.id || generateId("ex"),
        landmarks: ex.landmarks,
        thumbnail: ex.thumbnail || null,
        capturedAt: ex.capturedAt || now
      }))
    }));
    project._migratedFromLegacySigns = legacySigns.length > 0;
    delete project.signs;
  }

  project.testHistory = Array.isArray(project.testHistory) ? project.testHistory : [];
  project.settings = project.settings || {};
  project.settings.k = project.settings.k || 5;
  project.settings.confidenceThreshold = project.settings.confidenceThreshold ?? 65;
  project.settings.dynamicK = project.settings.dynamicK || 3;
  project.settings.similarityThreshold = project.settings.similarityThreshold ?? 60;
  project.settings.autoAddEnabled = project.settings.autoAddEnabled ?? true;
  project.settings.stableFrames = project.settings.stableFrames || 6;
  project.settings.motionSensitivity = project.settings.motionSensitivity ?? 1;
  project.stats = project.stats || { recognitions: 0, confidenceSum: 0 };

  // Versión 2 del reconocedor: la similitud ahora se mide contra la variación de los propios
  // ejemplos de cada seña (antes era una distancia fija), así que los umbrales anteriores ya
  // no significan lo mismo y se actualizan. Las señas dinámicas grabadas antes no guardaron la
  // trayectoria de las manos: se conservan, pero quedan "sin entrenar" hasta volver a grabarlas.
  if ((project.settings.recognizerVersion || 1) < 2) {
    project.settings.confidenceThreshold = 65;
    project.settings.similarityThreshold = 60;
    project.settings.recognizerVersion = 2;
    project.patterns.forEach((p) => {
      if (p.type === "dynamic" && countLegacyDynamicExamples(p) > 0) p.trained = false;
    });
  }

  return project;
}

/** Cuenta los ejemplos dinámicos grabados sin trayectoria (versión anterior de la aplicación). */
export function countLegacyDynamicExamples(pattern) {
  if (pattern.type !== "dynamic") return 0;
  return pattern.examples.filter((ex) => !Array.isArray(ex.frames) || !ex.frames.every((f) => Array.isArray(f.pos))).length;
}

/** Crea un nuevo proyecto vacío y lo guarda en la base de datos. */
export async function createProject(name) {
  await openDatabase();
  const now = new Date().toISOString();
  const project = migrateProjectShape({
    id: generateId("proj"),
    name: name && name.trim() ? name.trim() : "Proyecto sin nombre",
    createdAt: now,
    updatedAt: now,
    patterns: []
  });
  await putProject(project);
  return project;
}

/** Guarda (crea o actualiza) un proyecto completo. */
export function putProject(project) {
  return new Promise(async (resolve, reject) => {
    const db = await openDatabase();
    project.updatedAt = new Date().toISOString();
    const tx = db.transaction(STORE_NAME, "readwrite");
    tx.objectStore(STORE_NAME).put(project);
    tx.oncomplete = () => resolve(project);
    tx.onerror = () => reject(new Error("No se pudo guardar el proyecto."));
  });
}

/** Devuelve todos los proyectos guardados (migrados si hacía falta), ordenados por fecha de modificación. */
export async function getAllProjects() {
  const db = await openDatabase();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readonly");
    const request = tx.objectStore(STORE_NAME).getAll();
    request.onsuccess = () => {
      const projects = (request.result || []).map(migrateProjectShape);
      projects.sort((a, b) => new Date(b.updatedAt) - new Date(a.updatedAt));
      resolve(projects);
    };
    request.onerror = () => reject(new Error("No se pudieron leer los proyectos guardados."));
  });
}

/** Obtiene un proyecto por su id (migrado si hacía falta). */
export async function getProject(id) {
  const db = await openDatabase();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readonly");
    const request = tx.objectStore(STORE_NAME).get(id);
    request.onsuccess = () => resolve(request.result ? migrateProjectShape(request.result) : null);
    request.onerror = () => reject(new Error("No se pudo abrir el proyecto."));
  });
}

/** Elimina un proyecto por su id. */
export async function deleteProject(id) {
  const db = await openDatabase();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    tx.objectStore(STORE_NAME).delete(id);
    tx.oncomplete = () => resolve(true);
    tx.onerror = () => reject(new Error("No se pudo eliminar el proyecto."));
  });
}

/** Renombra un proyecto. */
export async function renameProject(id, newName) {
  const project = await getProject(id);
  if (!project) throw new Error("Proyecto no encontrado.");
  project.name = newName.trim() || project.name;
  return putProject(project);
}

/* ---------------------------------------------------------------------
   Utilidades sobre PATRONES (señas estáticas, dinámicas, secuencias,
   expresiones compuestas) dentro de un proyecto
   --------------------------------------------------------------------- */

export const PATTERN_TYPES = {
  static: { label: "Seña estática", needsCamera: true },
  dynamic: { label: "Seña dinámica", needsCamera: true },
  sequence: { label: "Secuencia de señas", needsCamera: false },
  composite: { label: "Expresión compuesta", needsCamera: false }
};

export function findPattern(project, name) {
  const normalized = name.trim().toUpperCase();
  return project.patterns.find((p) => p.name === normalized) || null;
}

/** Crea un nuevo patrón (o devuelve el existente si ya hay uno con ese nombre). */
export function createOrGetPattern(project, { name, type, description, source }) {
  const normalized = name.trim().toUpperCase();
  let pattern = project.patterns.find((p) => p.name === normalized);
  if (pattern) return pattern;

  const now = new Date().toISOString();
  pattern = {
    id: generateId("pat"),
    name: normalized,
    type: type || "static",
    description: (description || "").trim(),
    source: (source || "").trim(),
    trained: false,
    trainedAt: null,
    createdAt: now,
    updatedAt: now,
    examples: [],       // para "static" y "dynamic"
    sequenceUnits: []    // para "sequence" y "composite": lista ordenada de nombres de patrones
  };
  project.patterns.push(pattern);
  return pattern;
}

/** Agrega un ejemplo ESTÁTICO (un único vector de landmarks) a un patrón. */
export function addStaticExample(pattern, landmarks, thumbnail) {
  pattern.examples.push({
    id: generateId("ex"),
    landmarks,
    thumbnail: thumbnail || null,
    capturedAt: new Date().toISOString()
  });
  pattern.trained = false; // un ejemplo nuevo deja el patrón desactualizado hasta volver a entrenar
  pattern.updatedAt = new Date().toISOString();
}

/**
 * Agrega un ejemplo DINÁMICO (una secuencia de fotogramas con marca de
 * tiempo y vector de landmarks cada uno) a un patrón.
 * frames: [{ t: milisegundosDesdeInicio, vector: number[] }, ...]
 */
export function addDynamicExample(pattern, frames, thumbnail) {
  pattern.examples.push({
    id: generateId("ex"),
    frames,
    durationMs: frames.length > 0 ? frames[frames.length - 1].t : 0,
    frameCount: frames.length,
    thumbnail: thumbnail || null,
    capturedAt: new Date().toISOString()
  });
  pattern.trained = false;
  pattern.updatedAt = new Date().toISOString();
}

/** Elimina un ejemplo (estático o dinámico) de un patrón por su id. */
export function removeExample(pattern, exampleId) {
  pattern.examples = pattern.examples.filter((e) => e.id !== exampleId);
  pattern.trained = false;
  pattern.updatedAt = new Date().toISOString();
}

/** Define (o reemplaza) la lista ordenada de unidades de una secuencia o expresión compuesta. */
export function setSequenceUnits(pattern, orderedNames) {
  pattern.sequenceUnits = orderedNames.slice();
  pattern.trained = false;
  pattern.updatedAt = new Date().toISOString();
}

/** Marca un patrón como entrenado (listo para usarse en el reconocimiento). */
export function markPatternTrained(pattern) {
  pattern.trained = true;
  pattern.trainedAt = new Date().toISOString();
  pattern.updatedAt = pattern.trainedAt;
}

/** Elimina un patrón completo del proyecto. También lo quita de cualquier secuencia que lo referencie. */
export function removePattern(project, name) {
  project.patterns = project.patterns.filter((p) => p.name !== name);
  project.patterns.forEach((p) => {
    if (Array.isArray(p.sequenceUnits) && p.sequenceUnits.includes(name)) {
      p.sequenceUnits = p.sequenceUnits.filter((u) => u !== name);
      p.trained = false;
    }
  });
}

/** Registra un resultado de evaluación (prueba con un ejemplo nuevo, no usado en el entrenamiento). */
export function addTestResult(project, { expectedLabel, predictedLabel, mode, score }) {
  project.testHistory.push({
    id: generateId("test"),
    expectedLabel,
    predictedLabel: predictedLabel || null,
    correct: predictedLabel === expectedLabel,
    mode, // "static" | "dynamic"
    score: typeof score === "number" ? score : null,
    at: new Date().toISOString()
  });
  // Se conservan como máximo las últimas 200 pruebas para no crecer indefinidamente.
  if (project.testHistory.length > 200) {
    project.testHistory = project.testHistory.slice(-200);
  }
}

/** Calcula métricas reales de evaluación (aciertos, errores, matriz de confusión) a partir de testHistory. */
export function computeEvaluationMetrics(project) {
  const tests = project.testHistory || [];
  if (tests.length === 0) return null;

  const total = tests.length;
  const correct = tests.filter((t) => t.correct).length;
  const accuracy = Math.round((correct / total) * 100);

  const confusion = {}; // { esperado: { predicho: conteo } }
  tests.forEach((t) => {
    const expected = t.expectedLabel;
    const predicted = t.predictedLabel || "(sin reconocer)";
    confusion[expected] = confusion[expected] || {};
    confusion[expected][predicted] = (confusion[expected][predicted] || 0) + 1;
  });

  return { total, correct, accuracy, confusion };
}

/** Calcula estadísticas reales y verificables del proyecto (sin inventar datos). */
export function computeProjectStats(project) {
  const byType = { static: 0, dynamic: 0, sequence: 0, composite: 0 };
  let trainedCount = 0;
  let totalExamples = 0;

  project.patterns.forEach((p) => {
    byType[p.type] = (byType[p.type] || 0) + 1;
    if (p.trained) trainedCount++;
    totalExamples += (p.examples || []).length;
  });

  const recognitions = project.stats?.recognitions || 0;
  const confidenceSum = project.stats?.confidenceSum || 0;
  const avgConfidence = recognitions > 0 ? Math.round(confidenceSum / recognitions) : null;
  const evaluation = computeEvaluationMetrics(project);

  return { byType, trainedCount, totalPatterns: project.patterns.length, totalExamples, recognitions, avgConfidence, evaluation };
}

/* ---------------------------------------------------------------------
   Exportar / Importar proyectos como archivo JSON
   --------------------------------------------------------------------- */

/**
 * Genera y descarga un archivo JSON con el contenido del proyecto.
 * Por defecto NO incluye las miniaturas (thumbnail) de las capturas,
 * pero SÍ incluye siempre los landmarks y las secuencias de frames,
 * porque son los datos que el reconocedor necesita realmente para
 * volver a funcionar en otro computador. Exportar solo los nombres de
 * las señas dejaría un proyecto "vacío" en la práctica.
 */
export function exportProjectToFile(project, includeThumbnails = false) {
  const exportable = JSON.parse(JSON.stringify(project));
  if (!includeThumbnails) {
    exportable.patterns.forEach((pattern) => {
      pattern.examples.forEach((ex) => {
        ex.thumbnail = null;
      });
    });
  }
  exportable.exportedAt = new Date().toISOString();
  exportable.formatVersion = 2;

  const blob = new Blob([JSON.stringify(exportable, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  const safeName = (project.name || "proyecto").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "");
  a.href = url;
  a.download = `traductor-lsc-${safeName || "proyecto"}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/**
 * Lee un archivo JSON exportado previamente (formato nuevo con "patterns"
 * o formato antiguo con "signs") y lo guarda como un nuevo proyecto.
 */
export async function importProjectFromFile(file) {
  const text = await file.text();
  let data;
  try {
    data = JSON.parse(text);
  } catch (e) {
    throw new Error("El archivo importado no es un JSON válido.");
  }

  if (!data || typeof data !== "object" || (!Array.isArray(data.patterns) && !Array.isArray(data.signs))) {
    throw new Error("El archivo importado no tiene el formato esperado de un proyecto del Traductor LSC.");
  }

  const now = new Date().toISOString();
  const project = migrateProjectShape({
    id: generateId("proj"),
    name: (data.name || "Proyecto importado") + " (importado)",
    createdAt: data.createdAt || now,
    updatedAt: now,
    patterns: Array.isArray(data.patterns) ? data.patterns : undefined,
    signs: Array.isArray(data.signs) ? data.signs : undefined,
    testHistory: data.testHistory,
    settings: data.settings,
    stats: data.stats
  });

  await putProject(project);
  return project;
}
