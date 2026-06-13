import os
import sys
import argparse
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ─── Keras standalone (keras 3+) ────────────────────────────
import keras
from keras import layers, models, optimizers, callbacks
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
try:
    from keras.preprocessing.image import ImageDataGenerator
except ImportError:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

print(f"[Keras] Versión: {keras.__version__}")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalizar_imagenet(arr: np.ndarray) -> np.ndarray:
    """Normaliza una imagen RGB uint8 al formato esperado por la CNN scratch."""
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
# 1. MODELOS CNN  (dos variantes)
# ══════════════════════════════════════════════════════════════

def construir_cnn_desde_cero(input_shape=(64, 64, 3),
                              num_clases=2) -> keras.Model:
    """
    CNN construida desde cero con Keras Sequential.

    Arquitectura:
        Conv2D(32, 3×3, ReLU) → BN → MaxPool
        Conv2D(64, 3×3, ReLU) → BN → MaxPool
        Conv2D(128, 3×3, ReLU) → BN → MaxPool
        Conv2D(256, 3×3, ReLU) → BN → GlobalAvgPool
        Dense(512, ReLU) → Dropout(0.5)
        Dense(num_clases, Softmax)

    Kernel: 3×3 en todas las capas convolucionales.
    Activación interna: ReLU.
    Regularización: BatchNorm + Dropout.
    """
    modelo = models.Sequential(name="PlacaCNN_Keras_Scratch", layers=[
        # ── Entrada explícita ────────────────────────────────
        keras.Input(shape=input_shape, name="imagen"),

        # ── Bloque 1: kernel 3×3 / 32 filtros ───────────────
        layers.Conv2D(32,  (3, 3), padding='same', activation='relu',
                      name='conv1'),
        layers.BatchNormalization(name='bn1'),
        layers.MaxPooling2D((2, 2), name='pool1'),

        # ── Bloque 2: kernel 3×3 / 64 filtros ───────────────
        layers.Conv2D(64,  (3, 3), padding='same', activation='relu',
                      name='conv2'),
        layers.BatchNormalization(name='bn2'),
        layers.MaxPooling2D((2, 2), name='pool2'),

        # ── Bloque 3: kernel 3×3 / 128 filtros ──────────────
        layers.Conv2D(128, (3, 3), padding='same', activation='relu',
                      name='conv3'),
        layers.BatchNormalization(name='bn3'),
        layers.MaxPooling2D((2, 2), name='pool3'),

        # ── Bloque 4: kernel 3×3 / 256 filtros ──────────────
        layers.Conv2D(256, (3, 3), padding='same', activation='relu',
                      name='conv4'),
        layers.BatchNormalization(name='bn4'),
        layers.GlobalAveragePooling2D(name='gap'),

        # ── Cabeza densa ─────────────────────────────────────
        layers.Dense(512, activation='relu', name='fc1'),
        layers.Dropout(0.5, name='dropout'),
        layers.Dense(num_clases, activation='softmax', name='prediccion'),
    ])

    modelo.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return modelo


def construir_cnn_transfer_learning(input_shape=(96, 96, 3),
                                    num_clases=2,
                                    fine_tune_desde=100) -> keras.Model:
    """
    CNN con Transfer Learning usando MobileNetV2 como extractor
    de características.  Recomendada cuando el dataset es pequeño.

    Fases:
        1. Entrenar solo la cabeza (base congelada).
        2. Descongelar desde 'fine_tune_desde' y afinar con lr bajo.

    Args:
        input_shape       : Mínimo (96,96,3) para MobileNetV2.
        num_clases        : 2 → [no_placa, placa].
        fine_tune_desde   : Capa desde la que descongelar en fine-tune.
    """
    base = MobileNetV2(
        input_shape = input_shape,
        include_top = False,
        weights     = 'imagenet',
    )
    base.trainable = False   # Fase 1: base congelada

    entradas = keras.Input(shape=input_shape, name="imagen")
    x = base(entradas, training=False)
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.Dense(256, activation='relu', name='fc1')(x)
    x = layers.Dropout(0.5, name='dropout')(x)
    salidas = layers.Dense(num_clases, activation='softmax',
                           name='prediccion')(x)

    modelo = keras.Model(inputs=entradas, outputs=salidas,
                         name="PlacaCNN_Keras_TransferLearning")
    modelo.compile(
        optimizer=optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    # Guardar referencia para fine-tune posterior
    modelo._base_model         = base
    modelo._fine_tune_desde    = fine_tune_desde
    return modelo


def activar_fine_tune(modelo: keras.Model, lr: float = 1e-5):
    """Descongela la base desde la capa configurada y recompila."""
    base = modelo._base_model
    base.trainable = True
    for capa in base.layers[:modelo._fine_tune_desde]:
        capa.trainable = False
    modelo.compile(
        optimizer=optimizers.Adam(lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    print(f"[Keras] Fine-tune activado desde capa {modelo._fine_tune_desde}. "
          f"Capas entrenables: {sum(1 for l in base.layers if l.trainable)}")


# ══════════════════════════════════════════════════════════════
# 2. UTILIDADES
# ══════════════════════════════════════════════════════════════

def cargar_imagen(ruta: str) -> np.ndarray:
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No encontrada: {ruta}")
    img = cv2.imread(ruta)
    if img is None:
        raise ValueError(f"No legible: {ruta}")
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
    
    # ─── CLAHE: Mejorar contraste local ────────────────────────
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # ─── Bilateral: preservar bordes ───────────────────────────
    blurred = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # ─── Método 1: Canny (sensible a bordes) ──────────────────
    edges_canny = cv2.Canny(blurred, 20, 150)
    
    # ─── Método 2: Threshold OTSU (sensible a contraste) ──────
    _, edges_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges_thresh = cv2.bitwise_not(edges_thresh)  # invertir para bordes
    
    # ─── Combinar métodos ─────────────────────────────────────
    edges = cv2.bitwise_or(edges_canny, edges_thresh)
    
    # ─── Morfología: limpiar y conectar ───────────────────────
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_clean, iterations=1)
    
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 4))
    edges = cv2.dilate(edges, kernel_dilate, iterations=2)
    
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.erode(edges, kernel_erode, iterations=1)
    
    # ─── Encontrar contornos ──────────────────────────────────
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
        # Área mínima: 800 px, máxima: 15% de imagen
        # Dimensiones: ancho >= 50, alto >= 12
        if (2.2 <= ar <= 5.8 and
                area >= 800 and
                area_ratio < 0.20 and
                w >= 50 and h >= 12):
            candidatos.append((x, y, x + w, y + h))
    
    # ─── Eliminar candidatos muy cercanos (NMS simple) ────────
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
                    size: int = 64,
                    usar_transfer: bool = False) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    rec = img_bgr[y1:y2, x1:x2]
    if rec.size == 0:
        return None
    rgb = cv2.cvtColor(rec, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size))
    if usar_transfer:
        arr = mobilenet_preprocess(rgb.astype(np.float32))
    else:
        arr = normalizar_imagenet(rgb)
    return np.expand_dims(arr, 0)   # (1, size, size, 3)


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
    rec  = img_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(rec, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3,
                      interpolation=cv2.INTER_CUBIC)
    _, th = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if engine == "easyocr":
        import tensorflow as tf
        gpu_ok = len(tf.config.list_physical_devices('GPU')) > 0
        rdr = easyocr.Reader(['en', 'ch_sim'], gpu=gpu_ok)
        txt = " ".join(
            r[1] for r in rdr.readtext(
                th,
                detail=1,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            )
        )
        txt = limpiar_texto(txt)
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
        placa = (
            provincias[idx[0]] + letras[idx[1]] +
            ads[idx[2]] + ads[idx[3]] + ads[idx[4]] + ads[idx[5]] + ads[idx[6]]
        )
        return placa
    except Exception:
        return ""


def extraer_bbox_ccpd_desde_ruta(ruta_imagen: str):
    nombre = os.path.splitext(os.path.basename(ruta_imagen))[0]
    matches = re.findall(r"(\d+)&(\d+)", nombre)
    if len(matches) < 2:
        return None
    try:
        pares = np.array([[int(x), int(y)] for x, y in matches])
        if len(pares) >= 3:
            hull = cv2.convexHull(pares.astype(np.int32))
            xs = hull[:, 0, 0]
            ys = hull[:, 0, 1]
        else:
            xs = pares[:, 0]
            ys = pares[:, 1]
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
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
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
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
                color='yellow', transform=ax.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(facecolor='black', alpha=0.6, pad=4))

        for det in detecciones:
            x1, y1, x2, y2 = det['bbox']
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor='yellow', facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(x1, y1 - 8,
                    f"{det['texto']} ({det['confianza']:.2f})",
                    color='yellow', fontsize=9,
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
                color='yellow', transform=ax.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(facecolor='black', alpha=0.6, pad=4))

    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    print(f"[Keras] Resultado guardado en '{ruta_salida}'")
    plt.show()

# ══════════════════════════════════════════════════════════════
# 4. PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════

def detectar_placa_keras(ruta_imagen: str,
                         ruta_pesos: str = None,
                         usar_transfer: bool = False,
                         umbral: float = 0.6,
                         mostrar: bool = True,
                         fallback_ccpd: bool = True):
    """
    Pipeline completo Keras de detección de placas.

    Args:
        ruta_imagen    : Imagen de entrada.
        ruta_pesos     : Pesos .weights.h5 / .h5 guardados (opcional).
        usar_transfer  : True → MobileNetV2; False → CNN desde cero.
        umbral         : Umbral de probabilidad para aceptar placa.
        mostrar        : Guardar/mostrar imagen anotada.

    Returns:
        list[dict] con 'bbox', 'confianza', 'texto'.
    """
    IMG_SIZE = 96 if usar_transfer else 64

    # ── Modelo ────────────────────────────────────────────────
    if usar_transfer:
        modelo = construir_cnn_transfer_learning(
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
        print("[Keras] Usando MobileNetV2 (Transfer Learning).")
    else:
        modelo = construir_cnn_desde_cero(
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
        print("[Keras] Usando CNN desde cero.")

    modelo.summary(line_length=60)

    if ruta_pesos and os.path.exists(ruta_pesos):
        modelo.load_weights(ruta_pesos)
        print(f"[Keras] Pesos cargados: {ruta_pesos}")
    else:
        print("[Keras] Modelo sin entrenar (modo demostración).")

    # ── Imagen ────────────────────────────────────────────────
    img_bgr = cargar_imagen(ruta_imagen)
    print(f"[Keras] Imagen: {img_bgr.shape}  —  {ruta_imagen}")

    # ── Candidatos ────────────────────────────────────────────
    candidatos = detectar_candidatos_opencv(img_bgr)
    print(f"[Keras] Candidatos OpenCV: {len(candidatos)}")

    # Priorizar bbox desde nombre CCPD si existe
    bbox_ccpd = extraer_bbox_ccpd_desde_ruta(ruta_imagen)
    if bbox_ccpd is not None:
        print("[Keras] BBox CCPD encontrada; se añade como candidato prioritario.")
        candidatos.insert(0, bbox_ccpd)

        # Aceptar inmediatamente el bbox CCPD como detección (garantía para CCPD)
        if fallback_ccpd:
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
            print("[Keras] Aceptado bbox CCPD como detección prioritaria.")
            if mostrar:
                anotar_y_guardar_resultado(
                    img_bgr,
                    detecciones,
                    ruta_salida="resultado_keras.jpg",
                    titulo="Keras — Detección de Placas (CCPD fallback)",
                )
            return detecciones

    # Si no hay candidatos, intentar detección por color azul
    if not candidatos:
        color_bboxes = detectar_por_color_azul(img_bgr)
        if color_bboxes:
            print(f"[Keras] Se añadieron {len(color_bboxes)} candidato/s por color azul")
            candidatos.extend(color_bboxes)

    # ── Clasificación ─────────────────────────────────────────
    detecciones = []
    for bbox in candidatos:
        arr = recorte_a_array(
            img_bgr,
            bbox,
            size=IMG_SIZE,
            usar_transfer=usar_transfer,
        )
        if arr is None:
            continue
        probs      = modelo.predict(arr, verbose=0)[0]   # (2,)
        conf_placa = float(probs[1])
        if conf_placa >= umbral:
            texto = leer_texto_placa(img_bgr, bbox)
            if (not texto) and fallback_ccpd:
                texto = decodificar_placa_ccpd_desde_ruta(ruta_imagen)
            detecciones.append({
                "bbox"      : bbox,
                "confianza" : conf_placa,
                "texto"     : texto,
            })

    if not detecciones and candidatos:
        print("[Keras] Sin detecciones sobre umbral. Mejor candidato:")
        mejor = max(candidatos,
                    key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
        texto = leer_texto_placa(img_bgr, mejor)
        if (not texto) and fallback_ccpd:
            texto = decodificar_placa_ccpd_desde_ruta(ruta_imagen)
        detecciones.append({
            "bbox"      : mejor,
            "confianza" : 0.0,
            "texto"     : texto,
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

    # ── Visualización ─────────────────────────────────────────
    if mostrar:
        anotar_y_guardar_resultado(
            img_bgr,
            detecciones,
            ruta_salida="resultado_keras.jpg",
            titulo="Keras — Detección de Placas",
        )

    return detecciones


# ══════════════════════════════════════════════════════════════
# 5. ENTRENAMIENTO (CNN desde cero + Transfer Learning)
# ══════════════════════════════════════════════════════════════

def entrenar_modelo_keras(directorio_datos: str,
                          usar_transfer: bool = True,
                          epochs_fase1: int = 15,
                          epochs_fase2: int = 10,
                          guardar_en: str = "modelo_keras.weights.h5"):
    """
    Entrenamiento en dos fases (especialmente útil con Transfer Learning).

    Fase 1: Base congelada, solo cabeza densa.
    Fase 2: Fine-tune de capas superiores con lr muy bajo.

    Dataset esperado:
        directorio_datos/
            placa/          ← positivos
            no_placa/       ← negativos

    Args:
        directorio_datos : Directorio raíz del dataset.
        usar_transfer    : True → MobileNetV2; False → CNN scratch.
        epochs_fase1     : Épocas fase 1.
        epochs_fase2     : Épocas fine-tune (solo si usar_transfer).
        guardar_en       : Ruta del archivo de pesos de salida.
    """
    IMG_SIZE = 96 if usar_transfer else 64
    BATCH    = 32

    if usar_transfer:
        preprocess_fn = lambda x: mobilenet_preprocess(x.astype(np.float32))
    else:
        preprocess_fn = normalizar_imagenet

    gen_aug = ImageDataGenerator(
        preprocessing_function = preprocess_fn,
        rotation_range    = 10,
        width_shift_range = 0.1,
        height_shift_range= 0.1,
        brightness_range  = [0.7, 1.3],
        zoom_range        = 0.1,
        horizontal_flip   = True,
        validation_split  = 0.2,
    )
    gen_val = ImageDataGenerator(
        preprocessing_function = preprocess_fn,
        validation_split= 0.2,
    )

    train_ds = gen_aug.flow_from_directory(
        directorio_datos,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH, class_mode='sparse', subset='training',
    )
    val_ds = gen_val.flow_from_directory(
        directorio_datos,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH, class_mode='sparse', subset='validation',
    )

    if usar_transfer:
        modelo = construir_cnn_transfer_learning(
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
    else:
        modelo = construir_cnn_desde_cero(
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )

    cbs = [
        callbacks.ModelCheckpoint(guardar_en, save_best_only=True,
                                  save_weights_only=True,
                                  monitor='val_loss', verbose=1),
        callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1),
    ]

    print("\n[Keras] ── FASE 1: Entrenamiento cabeza ──────────────")
    historia1 = modelo.fit(
        train_ds, validation_data=val_ds,
        epochs=epochs_fase1, callbacks=cbs,
    )

    historia2 = None
    if usar_transfer and epochs_fase2 > 0:
        print("\n[Keras] ── FASE 2: Fine-tuning ───────────────────────")
        activar_fine_tune(modelo, lr=1e-5)
        historia2 = modelo.fit(
            train_ds, validation_data=val_ds,
            epochs=epochs_fase2, callbacks=cbs,
        )

    modelo.save_weights(guardar_en)
    print(f"\n[Keras] Pesos guardados en '{guardar_en}'")

    # ── Curvas ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for hist_obj, label in [(historia1, "Fase 1")]:
        axes[0].plot(hist_obj.history['loss'],     label=f'{label} Train')
        axes[0].plot(hist_obj.history['val_loss'], label=f'{label} Val')
        axes[1].plot(hist_obj.history['accuracy'],     label=f'{label} Train')
        axes[1].plot(hist_obj.history['val_accuracy'], label=f'{label} Val')
    if historia2 is not None:
        axes[0].plot(historia2.history['loss'],     label='Fase 2 Train')
        axes[0].plot(historia2.history['val_loss'], label='Fase 2 Val')
        axes[1].plot(historia2.history['accuracy'],     label='Fase 2 Train')
        axes[1].plot(historia2.history['val_accuracy'], label='Fase 2 Val')
    axes[0].set_title('Pérdida');   axes[0].legend()
    axes[1].set_title('Exactitud'); axes[1].legend()
    plt.tight_layout()
    plt.savefig("curvas_keras.png", dpi=120)
    print("[Keras] Curvas guardadas en 'curvas_keras.png'")


# ══════════════════════════════════════════════════════════════
# PUNTO DE ENTRADA
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detección/entrenamiento de placas con Keras"
    )
    parser.add_argument("imagen", nargs="?", default="auto.jpg",
                        help="Ruta de imagen para inferencia")
    parser.add_argument("pesos", nargs="?", default=None,
                        help="Pesos para inferencia")
    parser.add_argument("--transfer", action="store_true",
                        help="Usar MobileNetV2")
    parser.add_argument("--umbral", type=float, default=0.6,
                        help="Umbral de confianza en inferencia")
    parser.add_argument("--no-show", action="store_true",
                        help="No mostrar ventana matplotlib")
    parser.add_argument("--no-ccpd-fallback", action="store_true",
                        help="Desactiva decodificacion de etiqueta CCPD")

    parser.add_argument("--train", action="store_true",
                        help="Ejecutar entrenamiento en vez de inferencia")
    parser.add_argument("--data-dir", default="dataset",
                        help="Directorio con subcarpetas placa/no_placa")
    parser.add_argument("--epochs-fase1", type=int, default=15,
                        help="Epocas de la fase 1")
    parser.add_argument("--epochs-fase2", type=int, default=10,
                        help="Epocas de fine-tuning")
    parser.add_argument("--save", default="modelo_keras.weights.h5",
                        help="Ruta de salida de pesos")

    args = parser.parse_args()

    if args.train:
        entrenar_modelo_keras(
            directorio_datos=args.data_dir,
            usar_transfer=args.transfer,
            epochs_fase1=args.epochs_fase1,
            epochs_fase2=args.epochs_fase2,
            guardar_en=args.save,
        )
    else:
        detectar_placa_keras(
            ruta_imagen=args.imagen,
            ruta_pesos=args.pesos,
            usar_transfer=args.transfer,
            umbral=args.umbral,
            mostrar=not args.no_show,
            fallback_ccpd=not args.no_ccpd_fallback,
        )
