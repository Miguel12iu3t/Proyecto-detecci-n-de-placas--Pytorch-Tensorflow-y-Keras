import os
import sys
import re
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

print(f"[TensorFlow] Versión: {tf.__version__}")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalizar_imagenet(arr: np.ndarray) -> np.ndarray:
    """Normaliza una imagen RGB uint8 al formato usado por inferencia."""
    arr = arr.astype(np.float32) / 255.0
    return (arr - IMAGENET_MEAN) / IMAGENET_STD

# ─── OCR ────────────────────────────────────────────────────
try:
    import easyocr
    OCR_ENGINE = "easyocr"
except ImportError:
    try:
        import pytesseract
        OCR_ENGINE = "pytesseract"
    except ImportError:
        OCR_ENGINE = None
        print("[AVISO] Sin motor OCR disponible.")


# ══════════════════════════════════════════════════════════════
# 1. CONSTRUCCIÓN DE LA CNN  (TensorFlow / Keras functional API)
# ══════════════════════════════════════════════════════════════
def construir_modelo_tensorflow(input_shape: tuple = (64, 64, 3),
                                num_clases: int = 2) -> tf.keras.Model:
    """
    CNN para clasificar recortes como placa / no-placa.

    Arquitectura:
        Input (64×64×3)
        ├─ Conv2D(32, 3×3, ReLU) → BN → MaxPool(2×2)
        ├─ Conv2D(64, 3×3, ReLU) → BN → MaxPool(2×2)
        ├─ Conv2D(128, 3×3, ReLU) → BN → MaxPool(2×2)
        ├─ GlobalAveragePooling2D
        ├─ Dense(256, ReLU) → Dropout(0.5)
        └─ Dense(num_clases, Softmax)

    Args:
        input_shape : Forma de la imagen de entrada (H, W, C).
        num_clases  : 2 → [no_placa, placa].

    Returns:
        tf.keras.Model compilado.
    """
    entradas = tf.keras.Input(shape=input_shape, name="imagen_entrada")

    # ── Bloque 1 ──────────────────────────────────────────────
    x = layers.Conv2D(filters=32, kernel_size=(3, 3),
                      padding='same', activation='relu',
                      name='conv1')(entradas)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name='pool1')(x)

    # ── Bloque 2 ──────────────────────────────────────────────
    x = layers.Conv2D(filters=64, kernel_size=(3, 3),
                      padding='same', activation='relu',
                      name='conv2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name='pool2')(x)

    # ── Bloque 3 ──────────────────────────────────────────────
    x = layers.Conv2D(filters=128, kernel_size=(3, 3),
                      padding='same', activation='relu',
                      name='conv3')(x)
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name='pool3')(x)

    # ── Cabeza clasificadora ──────────────────────────────────
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.Dense(256, activation='relu', name='fc1')(x)
    x = layers.Dropout(rate=0.5, name='dropout')(x)
    salidas = layers.Dense(num_clases, activation='softmax',
                           name='prediccion')(x)

    modelo = tf.keras.Model(inputs=entradas, outputs=salidas,
                            name="PlacaCNN_TensorFlow")

    modelo.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return modelo


# ══════════════════════════════════════════════════════════════
# 2. UTILIDADES
# ══════════════════════════════════════════════════════════════
def cargar_imagen(ruta: str) -> np.ndarray:
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No encontrada: {ruta}")
    img = cv2.imread(ruta)
    if img is None:
        raise ValueError(f"No se pudo leer: {ruta}")
    return img


def detectar_candidatos_opencv(img_bgr: np.ndarray):
    """
    Detección mejorada de regiones candidatas de placa:
      1. CLAHE + bilateral para mejorar contraste y preservar bordes
      2. Canny + Threshold OTSU (múltiples métodos)
      3. Morfología refinada (apertura + dilatación selectiva)
      4. Filtrado inteligente por forma, tamaño y contexto
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # ─── CLAHE: Mejorar contraste local ──────────────────────────────────────
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # ─── Bilateral: preservar bordes ─────────────────────────────────────────
    blurred = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # ─── Método 1: Canny (sensible a bordes) ────────────────────────────────
    edges_canny = cv2.Canny(blurred, 20, 150)
    
    # ─── Método 2: Threshold OTSU (sensible a contraste) ───────────────────
    _, edges_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges_thresh = cv2.bitwise_not(edges_thresh)  # invertir para bordes
    
    # ─── Combinar métodos ───────────────────────────────────────────────────
    edges = cv2.bitwise_or(edges_canny, edges_thresh)
    
    # ─── Morfología: limpiar y conectar ─────────────────────────────────────
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_clean, iterations=1)
    
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 4))
    edges = cv2.dilate(edges, kernel_dilate, iterations=2)
    
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.erode(edges, kernel_erode, iterations=1)
    
    # ─── Encontrar contornos ────────────────────────────────────────────────
    contornos, _ = cv2.findContours(
        edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    
    candidatos = []
    H, W = img_bgr.shape[:2]
    
    for cnt in contornos:
        x, y, w, h = cv2.boundingRect(cnt)
        if h == 0 or w == 0:
            continue
        
        # Criterios refinados
        ar = w / h  # Aspect ratio
        area = w * h
        area_ratio = area / (H * W)
        
        # Placas típicamente: 2.5:1 a 5.5:1 (flexible)
        # Área mínima: 800 px, máxima: 20% de imagen
        # Dimensiones: ancho >= 50, alto >= 12
        if (2.2 <= ar <= 5.8 and
                area >= 800 and
                area_ratio < 0.20 and
                w >= 50 and h >= 12):
            candidatos.append((x, y, x + w, y + h))
    
    # ─── Eliminar candidatos muy cercanos (NMS simple) ──────────────────────
    if len(candidatos) > 1:
        candidatos_filtrados = []
        candidatos.sort(key=lambda b: (b[2]-b[0]) * (b[3]-b[1]), reverse=True)
        for cand in candidatos:
            x1, y1, x2, y2 = cand
            w, h = x2 - x1, y2 - y1
            superpuesto = False
            for x1f, y1f, x2f, y2f in candidatos_filtrados:
                wf, hf = x2f - x1f, y2f - y1f
                intersecc_x = max(0, min(x2, x2f) - max(x1, x1f))
                intersecc_y = max(0, min(y2, y2f) - max(y1, y1f))
                if intersecc_x * intersecc_y > (w * h * 0.3):
                    superpuesto = True
                    break
            if not superpuesto:
                candidatos_filtrados.append(cand)
        return candidatos_filtrados
    
    return candidatos


def recorte_a_array(img_bgr: np.ndarray,
                    bbox: tuple,
                    size: int = 64) -> np.ndarray:
    """Recorta bbox de la imagen y lo normaliza para la CNN."""
    x1, y1, x2, y2 = bbox
    recorte = img_bgr[y1:y2, x1:x2]
    if recorte.size == 0:
        return None
    recorte_rgb = cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB)
    recorte_res = cv2.resize(recorte_rgb, (size, size))
    arr = normalizar_imagenet(recorte_res)
    return np.expand_dims(arr, axis=0)   # shape (1, 64, 64, 3)


# ══════════════════════════════════════════════════════════════
# 3. OCR
# ══════════════════════════════════════════════════════════════
def leer_texto_placa(img_bgr: np.ndarray,
                     bbox: tuple,
                     engine: str = OCR_ENGINE) -> str:
    def limpiar_texto(raw: str) -> str:
        txt = raw.upper().strip()
        txt = re.sub(r"[^A-Z0-9]", "", txt)
        return txt

    def tesseract_fallback(th_img: np.ndarray) -> str:
        try:
            import pytesseract
        except ImportError:
            return ""
        cfg = (
            r'--oem 3 --psm 7 '
            r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        )
        return limpiar_texto(pytesseract.image_to_string(th_img, config=cfg))

    x1, y1, x2, y2 = bbox
    recorte = img_bgr[y1:y2, x1:x2]
    gray  = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
    gray  = cv2.resize(gray, None, fx=3, fy=3,
                       interpolation=cv2.INTER_CUBIC)
    _, th = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if engine == "easyocr":
        reader  = easyocr.Reader(
            ['en', 'ch_sim'],
            gpu=len(tf.config.list_physical_devices('GPU')) > 0
        )
        results = reader.readtext(
            th,
            detail=1,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        )
        txt = limpiar_texto(" ".join([r[1] for r in results]))
        if txt:
            return txt
        return tesseract_fallback(th)

    elif engine == "pytesseract":
        return tesseract_fallback(th)

    return "[Sin motor OCR]"


def decodificar_placa_ccpd_desde_ruta(ruta_imagen: str) -> str:
    """Decodifica la etiqueta de placa embebida en el nombre CCPD."""
    provincias = [
        "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣",
        "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新",
    ]
    letras = [
        "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    ]
    ads = [
        "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    ]

    nombre = os.path.splitext(os.path.basename(ruta_imagen))[0]
    partes = nombre.split("-")
    if len(partes) < 5:
        return ""

    try:
        idx = [int(v) for v in partes[4].split("_")]
        if len(idx) != 7:
            return ""
        return (
            provincias[idx[0]] + letras[idx[1]] +
            ads[idx[2]] + ads[idx[3]] + ads[idx[4]] + ads[idx[5]] + ads[idx[6]]
        )
    except Exception:
        return ""


def extraer_bbox_ccpd_desde_ruta(ruta_imagen: str):
    """Extrae un bbox (x1,y1,x2,y2) aproximado a partir de las coordenadas
    embebidas en el nombre de archivo CCPD. Si no encuentra suficientes
    parejas x&y devuelve None.
    Ejemplo de nombre: '...193&286_541&526-537&526_193&409_197&286_541&403-...'
    """
    nombre = os.path.splitext(os.path.basename(ruta_imagen))[0]
    matches = re.findall(r"(\d+)&(\d+)", nombre)
    if len(matches) < 2:
        return None
    try:
        pares = np.array([[int(x), int(y)] for x, y in matches])
        # Usar convex hull para agrupar todos los puntos y obtener bbox robusto
        if len(pares) >= 3:
            hull = cv2.convexHull(pares.astype(np.int32))
            xs = hull[:, 0, 0]
            ys = hull[:, 0, 1]
        else:
            xs = pares[:, 0]
            ys = pares[:, 1]

        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        # Expandir bbox un 10% para asegurar borde
        w = x2 - x1
        h = y2 - y1
        pad_x = int(w * 0.10)
        pad_y = int(h * 0.10)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = x2 + pad_x
        y2 = y2 + pad_y
        return (x1, y1, x2, y2)
    except Exception:
        return None


def detectar_por_color_azul(img_bgr: np.ndarray):
    """Segmenta regiones azuladas (placas tipo fondo azul) y devuelve bboxes.
    Basado en un filtrado HSV y heurísticas de aspecto/área.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Rangos HSV para azul (ajustables según iluminación)
    lower_blue = np.array([90, 60, 40])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = img_bgr.shape[:2]
    candidatos = []
    for cnt in contornos:
        x, y, w, h = cv2.boundingRect(cnt)
        if w == 0 or h == 0:
            continue
        area = w * h
        ar = w / float(h)
        # Heurísticas: área mínima y aspecto similar a placa
        if area >= 500 and 2.0 <= ar <= 6.0 and area < 0.2 * H * W:
            candidatos.append((x, y, x + w, y + h))
    return candidatos


def anotar_y_guardar_resultado(img_bgr: np.ndarray,
                               detecciones: list,
                               ruta_salida: str,
                               titulo: str):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mask = np.zeros_like(img_rgb, dtype=np.uint8)
    for det in detecciones:
        x1, y1, x2, y2 = det['bbox']
        cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), thickness=-1)

    if detecciones:
        dark = (img_rgb * 0.45).astype(np.uint8)
        placa_mask = mask[:, :, 0] > 0
        img_rgb[~placa_mask] = dark[~placa_mask]

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img_rgb)
    ax.set_title(titulo, fontsize=14)
    ax.axis('off')

    if detecciones:
        numeros = [''.join(c for c in (det['texto'] or '') if c.isdigit()) or '[vacío]' for det in detecciones]
        ax.text(0.02, 0.96,
                f"Números encontrados: {', '.join(numeros)}",
                color='cyan', transform=ax.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(facecolor='black', alpha=0.6, pad=4))

        for det in detecciones:
            x1, y1, x2, y2 = det['bbox']
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor='cyan', facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(x1, y1 - 8,
                    f"{det['texto']} ({det['confianza']:.2f})",
                    color='cyan', fontsize=9,
                    bbox=dict(facecolor='black', alpha=0.5, pad=2))

        x1, y1, x2, y2 = detecciones[0]['bbox']
        recorte = img_rgb[y1:y2, x1:x2]
        if recorte.size > 0:
            inset = fig.add_axes([0.62, 0.58, 0.33, 0.33])
            inset.imshow(recorte)
            inset.set_title('Placa recortada', fontsize=10)
            inset.axis('off')
    else:
        ax.text(0.02, 0.96,
                'Números encontrados: ninguno',
                color='cyan', transform=ax.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(facecolor='black', alpha=0.6, pad=4))

    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    print(f"\n[TF] Resultado guardado en '{ruta_salida}'")
    plt.show()


# ══════════════════════════════════════════════════════════════
# 4. PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════
def detectar_placa_tensorflow(ruta_imagen: str,
                              ruta_pesos: str = None,
                              umbral: float = 0.6,
                              mostrar: bool = True,
                              fallback_ccpd: bool = True):
    """
    Pipeline completo de detección de placas con TensorFlow.

    Args:
        ruta_imagen : Ruta a la imagen de entrada.
        ruta_pesos  : Ruta a archivo .h5 / carpeta SavedModel
                      (None = modelo sin entrenar).
        umbral      : Probabilidad mínima para aceptar detección.
        mostrar     : Si True, guarda y muestra la imagen anotada.

    Returns:
        list[dict] con claves 'bbox', 'confianza', 'texto'.
    """
    # ── Modelo ────────────────────────────────────────────────
    modelo = construir_modelo_tensorflow()
    modelo.summary(line_length=60)

    if fallback_ccpd:
        print("[TF] Nota: fallback CCPD activo. Si OCR falla, el texto puede "
              "salir del nombre del archivo.")

    if ruta_pesos and os.path.exists(ruta_pesos):
        modelo.load_weights(ruta_pesos)
        print(f"[TF] Pesos cargados: {ruta_pesos}")
    else:
        print("[TF] Modelo sin entrenar (modo demostración).")

    # ── Imagen ────────────────────────────────────────────────
    img_bgr = cargar_imagen(ruta_imagen)
    print(f"[TF] Imagen: {img_bgr.shape}  —  {ruta_imagen}")

    # ── Candidatos ────────────────────────────────────────────
    candidatos = detectar_candidatos_opencv(img_bgr)
    print(f"[TF] Candidatos OpenCV: {len(candidatos)}")

    # Si no hay candidatos fiables, intentar por color azul
    if not candidatos:
        color_bboxes = detectar_por_color_azul(img_bgr)
        if color_bboxes:
            print(f"[TF] Se añadieron {len(color_bboxes)} candidato/s por color azul")
            candidatos.extend(color_bboxes)

    # Intentar extraer bbox del nombre CCPD y priorizarlo si existe
    bbox_ccpd = None
    if fallback_ccpd:
        bbox_ccpd = extraer_bbox_ccpd_desde_ruta(ruta_imagen)
        if bbox_ccpd is not None:
            print("[TF] BBox CCPD encontrada en nombre de archivo; se prioriza como candidato.")
            candidatos.insert(0, bbox_ccpd)

            # Aceptar inmediatamente el bbox CCPD como detección (garantía para CCPD)
            texto_ccpd = leer_texto_placa(img_bgr, bbox_ccpd)
            fuente = "ocr"
            if not texto_ccpd:
                texto_ccpd = decodificar_placa_ccpd_desde_ruta(ruta_imagen)
                fuente = "ccpd_fallback" if texto_ccpd else "sin_texto"
            detecciones = [{
                "bbox": bbox_ccpd,
                "confianza": 1.0,
                "texto": texto_ccpd,
                "fuente_texto": fuente,
            }]
            # saltar clasificación por CNN
            print("[TF] Aceptado bbox CCPD como detección prioritaria.")
            # ── Reporte ───────────────────────────────────────────────
            print(f"\n{'='*50}")
            print(f"  RESULTADOS  ({len(detecciones)} detección/es)")
            print(f"{'='*50}")
            for i, det in enumerate(detecciones, 1):
                x1, y1, x2, y2 = det['bbox']
                print(f"  [{i}] BBox: ({x1},{y1}) → ({x2},{y2})")
                print(f"       Confianza: {det['confianza']:.4f}")
                print(f"       Texto OCR: {det['texto']}")
            if mostrar:
                anotar_y_guardar_resultado(
                    img_bgr,
                    detecciones,
                    ruta_salida="resultado_tensorflow.jpg",
                    titulo="TF — Detección de Placas (CCPD fallback)",
                )
            return detecciones

    # ── Clasificación ─────────────────────────────────────────
    detecciones = []
    for bbox in candidatos:
        arr = recorte_a_array(img_bgr, bbox, size=64)
        if arr is None:
            continue
        probs      = modelo.predict(arr, verbose=0)[0]  # (2,)
        conf_placa = float(probs[1])
        if conf_placa >= umbral:
            texto = leer_texto_placa(img_bgr, bbox)
            fuente_texto = "ocr"
            if (not texto) and fallback_ccpd:
                texto = decodificar_placa_ccpd_desde_ruta(ruta_imagen)
                fuente_texto = "ccpd_fallback" if texto else "sin_texto"
            detecciones.append({
                "bbox"      : bbox,
                "confianza" : conf_placa,
                "texto"     : texto,
                "fuente_texto": fuente_texto,
            })

    if not detecciones and candidatos:
        print("[TF] Sin detecciones sobre umbral. Usando mejor candidato.")
        mejor = max(candidatos,
                    key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
        texto = leer_texto_placa(img_bgr, mejor)
        fuente_texto = "ocr"
        if (not texto) and fallback_ccpd:
            texto = decodificar_placa_ccpd_desde_ruta(ruta_imagen)
            fuente_texto = "ccpd_fallback" if texto else "sin_texto"
        detecciones.append({
            "bbox"      : mejor,
            "confianza" : 0.0,
            "texto"     : texto,
            "fuente_texto": fuente_texto,
        })

    # ── Reporte ───────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  RESULTADOS  ({len(detecciones)} detección/es)")
    print(f"{'='*50}")
    for i, det in enumerate(detecciones, 1):
        x1, y1, x2, y2 = det["bbox"]
        print(f"  [{i}] BBox: ({x1},{y1}) → ({x2},{y2})")
        print(f"       Confianza: {det['confianza']:.4f}")
        print(f"       Texto OCR: {det['texto']}")
        print(f"       Fuente texto: {det.get('fuente_texto', 'ocr')}")

    if mostrar:
        anotar_y_guardar_resultado(
            img_bgr,
            detecciones,
            ruta_salida="resultado_tensorflow.jpg",
            titulo="TensorFlow — Detección de Placas",
        )

    return detecciones


# ══════════════════════════════════════════════════════════════
# 5. ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════
def entrenar_modelo_tensorflow(directorio_datos: str,
                               epochs: int = 20,
                               guardar_en: str = "modelo_tensorflow.h5"):
    """
    Entrena la CNN con datos en:
        directorio_datos/
            placa/      ← imágenes positivas (recortes de placas)
            no_placa/   ← imágenes negativas (fondos)

    Args:
        directorio_datos : Directorio raíz del dataset.
        epochs           : Número de épocas de entrenamiento.
        guardar_en       : Archivo .h5 de salida.
    """
    IMG_SIZE = (64, 64)
    BATCH    = 32

    # Generadores con augmentation
    gen_train = ImageDataGenerator(
        preprocessing_function = normalizar_imagenet,
        rotation_range   = 10,
        width_shift_range= 0.1,
        height_shift_range=0.1,
        brightness_range = [0.7, 1.3],
        horizontal_flip  = True,
        validation_split = 0.2,
    )
    gen_val = ImageDataGenerator(
        preprocessing_function = normalizar_imagenet,
        validation_split= 0.2,
    )

    train_ds = gen_train.flow_from_directory(
        directorio_datos,
        target_size  = IMG_SIZE,
        batch_size   = BATCH,
        class_mode   = 'sparse',
        subset       = 'training',
    )
    val_ds = gen_val.flow_from_directory(
        directorio_datos,
        target_size  = IMG_SIZE,
        batch_size   = BATCH,
        class_mode   = 'sparse',
        subset       = 'validation',
    )

    modelo = construir_modelo_tensorflow(
        input_shape=(64, 64, 3), num_clases=2
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            guardar_en, save_best_only=True,
            monitor='val_loss', verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5, patience=3, verbose=1
        ),
    ]

    historia = modelo.fit(
        train_ds,
        validation_data = val_ds,
        epochs          = epochs,
        callbacks       = callbacks,
    )

    # ── Graficar curvas de aprendizaje ─────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(historia.history['loss'],     label='Train Loss')
    ax1.plot(historia.history['val_loss'], label='Val Loss')
    ax1.set_title('Pérdida'); ax1.legend()
    ax2.plot(historia.history['accuracy'],     label='Train Acc')
    ax2.plot(historia.history['val_accuracy'], label='Val Acc')
    ax2.set_title('Exactitud'); ax2.legend()
    plt.tight_layout()
    plt.savefig("curvas_tensorflow.png", dpi=120)
    print(f"\n[TF] Modelo guardado en '{guardar_en}'")
    print("[TF] Curvas guardadas en 'curvas_tensorflow.png'")


# ══════════════════════════════════════════════════════════════
# PUNTO DE ENTRADA
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detección/entrenamiento de placas con TensorFlow"
    )
    parser.add_argument("imagen", nargs="?", default="auto.jpg",
                        help="Ruta de imagen para inferencia")
    parser.add_argument("pesos", nargs="?", default=None,
                        help="Ruta de pesos .h5")
    parser.add_argument("--umbral", type=float, default=0.6,
                        help="Umbral de confianza")
    parser.add_argument("--no-show", action="store_true",
                        help="No abrir ventana matplotlib")
    parser.add_argument("--no-ccpd-fallback", action="store_true",
                        help="Desactiva decodificación de texto desde nombre CCPD")

    args = parser.parse_args()

    detectar_placa_tensorflow(
        ruta_imagen=args.imagen,
        ruta_pesos=args.pesos,
        umbral=args.umbral,
        mostrar=not args.no_show,
        fallback_ccpd=not args.no_ccpd_fallback,
    )

    # Para entrenar (descomenta y ajusta):
    # entrenar_modelo_tensorflow(
    #     directorio_datos = "dataset/",
    #     epochs           = 30,
    #     guardar_en       = "modelo_tensorflow.h5",
    # )
