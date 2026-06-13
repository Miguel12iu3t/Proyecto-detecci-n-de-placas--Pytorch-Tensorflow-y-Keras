# Detección de Placas de Autos
### Tres implementaciones: PyTorch · TensorFlow · Keras

---

## Archivos del proyecto

| Archivo | Librería |
|---|---|
| `detector_placas_pytorch.py`    | PyTorch |
| `detector_placas_tensorflow.py` | TensorFlow |
| `detector_placas_keras.py`      | Keras (con Transfer Learning) |

---

## Correr codigos de cada libreria

# Pytorch
python detector_placas_pytorch.py "ruta/imagen_CCPD.jpg" "pesos_pytorch_ccpd.pth"

# Tensorflow
python detector_placas_tensorflow.py "ruta/imagen_CCPD.jpg" "modelo_tensorflow_ccpd.h5"

# Keras
python detector_placas_keras.py "ruta/imagen_CCPD.jpg" "modelo_keras_ccpd.weights.h5" --transfer

## Pipeline de detección (igual en los 3 scripts)

```
Imagen
  │
  ▼
OpenCV (escala de grises + bilateral + Canny + morfología)
  │  Detecta regiones candidatas por relación de aspecto (2:1 a 6:1)
  ▼
CNN (PyTorch / TensorFlow / Keras)
  │  Clasifica cada candidato: placa (1) vs no-placa (0)
  ▼
OCR (EasyOCR preferido, pytesseract como respaldo)
  │  Lee los caracteres de la región detectada
  ▼
Resultado visual (imagen anotada .jpg)
```

---

## Arquitecturas CNN

### PyTorch — `PlacaCNN`
| Capa | Configuración |
|---|---|
| Conv1 | in=3, out=16, kernel=3×3, padding=1 → BN → ReLU → MaxPool 2×2 |
| Conv2 | in=16, out=32, kernel=3×3, padding=1 → BN → ReLU → MaxPool 2×2 |
| Conv3 | in=32, out=64, kernel=3×3, padding=1 → BN → ReLU → MaxPool 2×2 |
| FC1 | 64×8×8 → 256, ReLU, Dropout 0.5 |
| FC2 | 256 → 2, Softmax |

### TensorFlow — Functional API
| Capa | Configuración |
|---|---|
| Conv1 | 32 filtros, kernel=3×3, same, ReLU → BN → MaxPool 2×2 |
| Conv2 | 64 filtros, kernel=3×3, same, ReLU → BN → MaxPool 2×2 |
| Conv3 | 128 filtros, kernel=3×3, same, ReLU → BN → MaxPool 2×2 |
| GAP | GlobalAveragePooling2D |
| FC1 | 256, ReLU, Dropout 0.5 |
| Salida | 2, Softmax |

### Keras — Dos variantes
**CNN desde cero (Sequential):**
| Capa | Configuración |
|---|---|
| Conv1 | 32 filtros, 3×3, ReLU → BN → MaxPool |
| Conv2 | 64 filtros, 3×3, ReLU → BN → MaxPool |
| Conv3 | 128 filtros, 3×3, ReLU → BN → MaxPool |
| Conv4 | 256 filtros, 3×3, ReLU → BN → GAP |
| FC1 | 512, ReLU, Dropout 0.5 |
| Salida | 2, Softmax |

**Transfer Learning (MobileNetV2):**
- Base: MobileNetV2 (pesos ImageNet, input 96×96)
- Fase 1: base congelada, solo cabeza densa
- Fase 2 (fine-tune): capas superiores descongeladas con lr=1e-5

---

## Estructura de datos esperada

```
dataset/
├── placa/          ← recortes de placas (positivos)
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
└── no_placa/       ← recortes de fondo (negativos)
    ├── fondo001.jpg
    ├── fondo002.jpg
    └── ...
```

**Generar automáticamente desde CCPD2019:** Ver paso 1️⃣ en la sección "Flujo de uso".

---

## 💡 Recomendaciones

- **Sin dataset propio**: Los scripts funcionan en modo demostración usando heurísticas OpenCV + OCR directo.
- **Dataset pequeño (< 500 imgs)**: Usa el modo **Transfer Learning** de Keras.
- **Dataset grande**: Cualquiera de las tres CNNs desde cero.
- **OCR**: EasyOCR soporta chino simplificado (`ch_sim`) y es ideal para placas asiáticas como la de la imagen de prueba.
- **GPU**: Todos los scripts la detectan automáticamente (CUDA para PyTorch/TF).

---

## 🔧 Parámetros ajustables

| Parámetro | Dónde ajustar | Descripción |
|---|---|---|
| `umbral_confianza` | `detectar_placa_*()` | P(placa) mínima aceptada (default 0.6) |
| `input_size` | Constructor CNN | Resolución de recortes (default 64×64) |
| `epochs` | `entrenar_*()` | Épocas de entrenamiento |
| `lr` | `entrenar_*()` | Tasa de aprendizaje (default 1e-3) |
| Canny thresholds | `detectar_candidatos_opencv()` | Sensibilidad de detección de bordes |

---

## 📤 Salidas generadas

| Archivo | Descripción |
|---|---|
| `resultado_pytorch.jpg`    | Imagen con BBoxes y texto OCR (verde) |
| `resultado_tensorflow.jpg` | Imagen anotada (cian) |
| `resultado_keras.jpg`      | Imagen anotada (amarillo) |
| `curvas_tensorflow.png`    | Gráficas de entrenamiento TF |
| `curvas_keras.png`         | Gráficas de entrenamiento Keras |
| `pesos_pytorch.pth`        | Pesos guardados PyTorch |
| `modelo_tensorflow.h5`     | Modelo guardado TF |
| `modelo_keras.weights.h5`  | Pesos guardados Keras |
