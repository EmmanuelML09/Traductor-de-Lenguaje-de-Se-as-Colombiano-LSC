/* =====================================================================
   camera.js
   ---------------------------------------------------------------------
   Módulo responsable de todo lo relacionado con la webcam:
   - Encender / apagar la cámara
   - Listar y cambiar entre varias cámaras disponibles
   - Capturar un fotograma como imagen (para las miniaturas de ejemplos)
   - Traducir los errores del navegador en mensajes claros para el usuario

   Este módulo no cambió en la actualización: la captura de secuencias
   dinámicas (recognition.js) reutiliza este mismo controlador para leer
   fotogramas sucesivos del <video>, no necesita una cámara aparte.
   ===================================================================== */

/** Error de cámara con un "tipo" identificable para mostrar mensajes claros. */
export class CameraError extends Error {
  constructor(type, message) {
    super(message);
    this.name = "CameraError";
    this.type = type; // "no-support" | "no-device" | "permission-denied" | "busy" | "unknown"
  }
}

/**
 * Controlador de cámara reutilizable. Se instancia una vez por cada
 * elemento <video> que necesite mostrar la webcam (traductor, entrenamiento,
 * modo demostración).
 */
export class CameraController {
  constructor(videoElement) {
    this.video = videoElement;
    this.stream = null;
    this.devices = [];
    this.currentDeviceIndex = 0;
  }

  get isActive() {
    return !!this.stream;
  }

  /** Comprueba compatibilidad básica del navegador antes de intentar nada. */
  static checkBrowserSupport() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new CameraError(
        "no-support",
        "Este navegador no permite acceder a la cámara. Usa una versión reciente de Chrome, Edge o Firefox."
      );
    }
    if (location.protocol !== "https:" && location.hostname !== "localhost" && location.hostname !== "127.0.0.1") {
      throw new CameraError(
        "no-support",
        "Los navegadores solo permiten usar la cámara en conexiones seguras (HTTPS) o en localhost. Abre esta página mediante un servidor local (localhost) o publícala con HTTPS."
      );
    }
  }

  /** Actualiza la lista interna de cámaras disponibles (requiere permiso ya concedido para ver etiquetas). */
  async refreshDeviceList() {
    const all = await navigator.mediaDevices.enumerateDevices();
    this.devices = all.filter((d) => d.kind === "videoinput");
    return this.devices;
  }

  /** Enciende la cámara. deviceId opcional para elegir una cámara concreta. */
  async start(deviceId) {
    CameraController.checkBrowserSupport();

    const constraints = {
      audio: false,
      video: deviceId
        ? { deviceId: { exact: deviceId } }
        : { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } }
    };

    try {
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      this.stopStreamOnly();
      this.stream = stream;
      this.video.srcObject = stream;
      await this.video.play().catch(() => {});
      await this.refreshDeviceList();
      return true;
    } catch (err) {
      throw this._mapError(err);
    }
  }

  /** Cambia a la siguiente cámara disponible (si existe más de una). */
  async switchCamera() {
    if (this.devices.length < 2) {
      throw new CameraError("no-device", "No se detectó otra cámara disponible en este equipo.");
    }
    this.currentDeviceIndex = (this.currentDeviceIndex + 1) % this.devices.length;
    const nextDevice = this.devices[this.currentDeviceIndex];
    return this.start(nextDevice.deviceId);
  }

  /** Detiene únicamente las pistas del stream, sin limpiar el elemento de video. */
  stopStreamOnly() {
    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }
  }

  /** Apaga completamente la cámara. */
  stop() {
    this.stopStreamOnly();
    if (this.video) this.video.srcObject = null;
  }

  /**
   * Captura el fotograma actual del video como una miniatura JPEG (dataURL),
   * en baja resolución para no inflar el tamaño del proyecto guardado.
   */
  captureThumbnail(maxWidth = 160) {
    if (!this.video || this.video.videoWidth === 0) return null;
    const ratio = this.video.videoHeight / this.video.videoWidth;
    const canvas = document.createElement("canvas");
    canvas.width = maxWidth;
    canvas.height = Math.round(maxWidth * ratio);
    const ctx = canvas.getContext("2d");
    // Se voltea horizontalmente para que la miniatura coincida con lo que el usuario ve (efecto espejo).
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(this.video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.6);
  }

  /** Traduce los errores nativos de getUserMedia a mensajes claros para el usuario. */
  _mapError(err) {
    if (err instanceof CameraError) return err;
    const name = err && err.name;
    if (name === "NotAllowedError" || name === "SecurityError") {
      return new CameraError("permission-denied", "No se concedió permiso para usar la cámara. Revisa los permisos del navegador para este sitio.");
    }
    if (name === "NotFoundError" || name === "OverconstrainedError") {
      return new CameraError("no-device", "No se encontró ninguna cámara conectada a este equipo.");
    }
    if (name === "NotReadableError") {
      return new CameraError("busy", "La cámara parece estar siendo usada por otra aplicación. Ciérrala e inténtalo de nuevo.");
    }
    return new CameraError("unknown", "Ocurrió un error inesperado al intentar acceder a la cámara.");
  }
}
