"""Funciones auxiliares del proyecto 07MIAR (clasificación HAR).

Agrupa el código reutilizable del notebook para evitar repetir, en cada
experimento, la compilación, el entrenamiento, el registro de métricas y la
representación de las curvas de aprendizaje.
"""

import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib.ticker import MaxNLocator

# Colores de las series (entrenamiento / validación) usados en todas las figuras
COLOR_TRAIN = "#2a78d6"
COLOR_VAL = "#eb6834"


# ---------------------------------------------------------------------------
# Carga de imágenes
# ---------------------------------------------------------------------------
def cargar_imagenes(directorio, nombres_fichero, img_size):
    """Lee, convierte a RGB y redimensiona las imágenes indicadas.

    Devuelve un array uint8 (N, alto, ancho, 3) con las imágenes que se han
    podido leer y la lista de posiciones (en `nombres_fichero`) que han fallado,
    para que el llamador pueda eliminarlas del DataFrame antes de dividir.
    """
    imagenes, fallidas = [], []
    for pos, nombre in enumerate(nombres_fichero):
        img = cv2.imread(os.path.join(directorio, nombre))
        if img is None:
            fallidas.append(pos)
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imagenes.append(cv2.resize(img, img_size))
    return np.array(imagenes, dtype=np.uint8), fallidas


# ---------------------------------------------------------------------------
# Curvas de aprendizaje
# ---------------------------------------------------------------------------
def impresion(epochs, history_dict, titulo=None):
    """Representa accuracy y loss de entrenamiento y validación por época."""
    epochs = list(epochs)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4.5))

    paneles = [
        (ax1, "accuracy", "Evolución de la precisión (accuracy)", "Precisión"),
        (ax2, "loss", "Evolución de la pérdida (loss)", "Pérdida"),
    ]
    marcador = "o" if len(epochs) <= 30 else None  # sin marcadores en curvas largas
    for ax, clave, titulo_panel, ylabel in paneles:
        ax.plot(epochs, history_dict[clave], marker=marcador, markersize=4,
                linewidth=2, label="Entrenamiento", color=COLOR_TRAIN)
        ax.plot(epochs, history_dict["val_" + clave], marker=marcador, markersize=4,
                linewidth=2, label="Validación", color=COLOR_VAL)
        ax.set_title(titulo_panel, fontsize=12, weight="bold")
        ax.set_xlabel("Épocas", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        # Ticks enteros y en número limitado, aunque se entrenen 60-80 épocas
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=12))
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()

    # Época con mejor val_accuracy, como referencia visual
    mejor = int(np.argmax(history_dict["val_accuracy"]))
    ax1.axvline(epochs[mejor], color="grey", linestyle=":", linewidth=1)

    if titulo:
        fig.suptitle(titulo, fontsize=13, weight="bold")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Registro de resultados
# ---------------------------------------------------------------------------
COLUMNAS_REGISTRO = [
    "estrategia", "experimento", "best_val_accuracy", "best_epoch",
    "val_loss_best_epoch", "min_val_loss", "train_acc_best_epoch",
    "gap_acc", "epochs_entrenadas", "parametros", "tiempo_s", "s_por_epoca",
    "ruta_modelo",
]


def registrar_resultado(resultados, estrategia, experimento, history, modelo,
                        tiempo_s, ruta_modelo=None, **extra):
    """Añade (o sustituye) una fila con las métricas de validación de un entrenamiento.

    Es genérica: sirve tanto para los modelos from scratch como para los
    preentrenados. `history` puede ser un objeto History de Keras o su
    diccionario `history.history`. Las métricas se obtienen siempre del
    conjunto de VALIDACIÓN (nunca del test). Devuelve la fila añadida.
    """
    h = history.history if hasattr(history, "history") else history
    val_acc = np.asarray(h["val_accuracy"], dtype=float)
    mejor = int(np.argmax(val_acc))
    fila = {
        "estrategia": estrategia,
        "experimento": experimento,
        "best_val_accuracy": float(val_acc[mejor]),
        "best_epoch": mejor + 1,
        "val_loss_best_epoch": float(h["val_loss"][mejor]),
        "min_val_loss": float(np.min(h["val_loss"])),
        "train_acc_best_epoch": float(h["accuracy"][mejor]),
        "gap_acc": float(h["accuracy"][mejor] - val_acc[mejor]),
        "epochs_entrenadas": len(val_acc),
        "parametros": int(modelo.count_params()),
        "tiempo_s": round(float(tiempo_s), 1),
        "s_por_epoca": round(float(tiempo_s) / max(len(val_acc), 1), 1),
        "ruta_modelo": ruta_modelo,
    }
    fila.update(extra)
    # Si el experimento se vuelve a ejecutar, se sustituye su fila anterior
    resultados[:] = [r for r in resultados
                     if not (r["estrategia"] == estrategia and r["experimento"] == experimento)]
    resultados.append(fila)
    return fila


def tabla_resultados(resultados, estrategia=None, ordenar_por="best_val_accuracy"):
    """Convierte el registro (lista de dicts) en un DataFrame ordenado."""
    df = pd.DataFrame(resultados)
    if df.empty:
        return pd.DataFrame(columns=COLUMNAS_REGISTRO)
    if estrategia is not None:
        df = df[df["estrategia"] == estrategia]
    columnas = [c for c in COLUMNAS_REGISTRO if c in df.columns] + \
               [c for c in df.columns if c not in COLUMNAS_REGISTRO]
    return df[columnas].sort_values(ordenar_por, ascending=False).reset_index(drop=True)


def graficar_resultados(df, metrica="best_val_accuracy", titulo=None, referencia=None):
    """Diagrama de barras horizontal de una métrica de validación por experimento."""
    df = df.sort_values(metrica, ascending=True)
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(df) + 1.5))
    barras = ax.barh(df["experimento"], df[metrica], color=COLOR_TRAIN, height=0.6)
    ax.bar_label(barras, fmt="%.3f", padding=3, fontsize=9)
    if referencia is not None:
        ax.axvline(referencia, color="grey", linestyle="--", linewidth=1,
                   label=f"Clasificador aleatorio ({referencia:.3f})")
        ax.legend(loc="lower right")
    ax.set_xlabel(metrica)
    ax.set_xlim(0, min(1.0, max(df[metrica].max(), referencia or 0.0) * 1.15))
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.set_title(titulo or f"Comparación de experimentos ({metrica})",
                 fontsize=12, weight="bold")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Entrenamiento de un experimento
# ---------------------------------------------------------------------------
def entrenar_experimento(nombre, modelo, X_tr, y_tr, X_val, y_val, epochs,
                         batch_size, optimizer, callbacks=None, resultados=None,
                         historiales=None, estrategia="From scratch",
                         models_dir="modelos", loss="sparse_categorical_crossentropy",
                         mostrar_curvas=True, verbose=2):
    """Compila, entrena, registra y guarda un experimento.

    - Valida SIEMPRE sobre (X_val, y_val), el mismo conjunto para todos los
      experimentos, de modo que los resultados son comparables.
    - Guarda en `models_dir/<nombre>.keras` los pesos de la época con mayor
      val_accuracy (ModelCheckpoint; no altera el entrenamiento).
    - Mide el tiempo de entrenamiento, imprime un resumen de una línea, añade
      una fila al registro `resultados` y guarda `history.history` en
      `historiales[nombre]`.
    Devuelve el objeto History.
    """
    os.makedirs(models_dir, exist_ok=True)
    ruta_modelo = os.path.join(models_dir, f"{nombre}.keras")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        ruta_modelo, monitor="val_accuracy", mode="max",
        save_best_only=True, verbose=0,
    )

    modelo.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"],
                   jit_compile=False)

    inicio = time.perf_counter()
    history = modelo.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=list(callbacks or []) + [checkpoint],
        verbose=verbose,
    )
    tiempo_s = time.perf_counter() - inicio

    if historiales is not None:
        historiales[nombre] = history.history
    if resultados is not None:
        fila = registrar_resultado(resultados, estrategia, nombre, history,
                                   modelo, tiempo_s, ruta_modelo=ruta_modelo)
    else:
        fila = registrar_resultado([], estrategia, nombre, history, modelo,
                                   tiempo_s, ruta_modelo=ruta_modelo)

    print(f"[{nombre}] best val_accuracy={fila['best_val_accuracy']:.4f} "
          f"(época {fila['best_epoch']}/{fila['epochs_entrenadas']}) | "
          f"min val_loss={fila['min_val_loss']:.4f} | "
          f"train_acc en esa época={fila['train_acc_best_epoch']:.4f} | "
          f"parámetros={fila['parametros']:,} | tiempo={fila['tiempo_s']:.0f} s "
          f"({fila['s_por_epoca']:.1f} s/época)")

    if mostrar_curvas:
        h = history.history
        impresion(range(1, len(h["accuracy"]) + 1), h, titulo=nombre)
    return history


# ---------------------------------------------------------------------------
# Diagnóstico de neuronas muertas
# ---------------------------------------------------------------------------
def _tipo_activacion(layer):
    """Devuelve 'relu', 'leaky_relu' o None según la activación de la capa."""
    if isinstance(layer, tf.keras.layers.LeakyReLU):
        return "leaky_relu"
    if isinstance(layer, tf.keras.layers.ReLU):
        pendiente = float(getattr(layer, "negative_slope", 0.0) or 0.0)
        return "relu" if pendiente == 0.0 else "leaky_relu"
    act = getattr(layer, "activation", None)  # Conv2D, Dense y Activation
    nombre = getattr(act, "__name__", "")
    if nombre == "relu":
        return "relu"
    if nombre in ("leaky_relu", "LeakyReLU"):
        return "leaky_relu"
    return None


def diagnosticar_neuronas_muertas(modelo, datos_muestra):
    """Cuenta neuronas/filtros que no se activan para ninguna imagen de la muestra.

    Considera capas Conv2D/Dense con activation='relu' y capas separadas
    `ReLU`, `Activation('relu')` y `LeakyReLU`. Para ReLU, una unidad está
    "muerta" si su salida es 0 en todas las imágenes y posiciones; para
    LeakyReLU (que nunca anula el gradiente) se informa de las unidades que
    nunca producen salida positiva ("inactivas"). Devuelve un DataFrame.
    """
    capas = [(layer, _tipo_activacion(layer)) for layer in modelo.layers]
    capas = [(layer, tipo) for layer, tipo in capas if tipo is not None]
    if not capas:
        print("El modelo no contiene capas con activación ReLU/LeakyReLU.")
        return pd.DataFrame()

    entradas = modelo.inputs[0] if len(modelo.inputs) == 1 else modelo.inputs
    extractor = tf.keras.Model(inputs=entradas,
                               outputs=[layer.output for layer, _ in capas])
    salidas = extractor(datos_muestra, training=False)
    if not isinstance(salidas, (list, tuple)):
        salidas = [salidas]

    filas = []
    for (layer, tipo), act in zip(capas, salidas):
        act = np.asarray(act)
        ejes = tuple(range(act.ndim - 1))  # todos menos el de canales/neuronas
        # ReLU: salida siempre 0 (muerta); LeakyReLU: nunca positiva (inactiva)
        muertas = np.all(act <= 0, axis=ejes)
        total = muertas.size
        filas.append({
            "capa": layer.name,
            "activacion": tipo,
            "unidades": total,
            "muertas_o_inactivas": int(muertas.sum()),
            "porcentaje": round(100 * muertas.sum() / total, 1),
        })
    df = pd.DataFrame(filas)
    print(df.to_string(index=False))
    return df
