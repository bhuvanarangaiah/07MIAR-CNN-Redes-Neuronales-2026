import matplotlib.pyplot as plt

def impresion(epochs, history_dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))

    ax1.plot(
        epochs, history_dict["accuracy"], "o-", label="Entrenamiento", color="blue"
    )
    ax1.plot(
        epochs,
        history_dict["val_accuracy"],
        "s-",
        label="Validación",
        color="orange",
    )
    ax1.set_title("Evolución de la Precisión (Accuracy)", fontsize=13, weight="bold")
    ax1.set_xlabel("Épocas", fontsize=11)
    ax1.set_ylabel("Precisión", fontsize=11)
    ax1.set_xticks(epochs)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    ax2.plot(epochs, history_dict["loss"], "o-", label="Entrenamiento", color="blue")
    ax2.plot(
        epochs, history_dict["val_loss"], "s-", label="Validación", color="orange"
    )
    ax2.set_title(
        "Evolución de la Pérdida (Loss)", fontsize=13, weight="bold"
    )
    ax2.set_xlabel("Épocas", fontsize=11)
    ax2.set_ylabel("Pérdida", fontsize=11)
    ax2.set_xticks(epochs)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.show()

import numpy as np
import tensorflow as tf

def diagnosticar_neuronas_muertas(modelo, datos_muestra):
    # Seleccionamos todas las capas que usan ReLU (Conv2D y Dense)
    capas_relu = [
        layer for layer in modelo.layers 
        if hasattr(layer, 'activation') and layer.activation == tf.keras.activations.relu
    ]
    
    # Creamos un modelo auxiliar que devuelve la salida de cada capa con ReLU
    extractor = tf.keras.Model(
        inputs=modelo.inputs,
        outputs=[layer.output for layer in capas_relu]
    )
    
    # Pasamos una muestra de imágenes (ej. 256 imágenes) sin activar Dropout ni Augmentation
    activaciones = extractor.predict(datos_muestra, verbose=0)
    
    print(f"{'Capa':<25} | {'Total':<8} | {'Muertas':<8} | {'% Muertas'}")
    print("-" * 55)
    
    for layer, act in zip(capas_relu, activaciones):
        if len(act.shape) == 4:  # Capa Conv2D: (batch, alto, ancho, filtros)
            # Revisamos si un filtro entero es 0 en todas las imágenes y píxeles
            muertas = np.all(act == 0, axis=(0, 1, 2))
        else:  # Capa Dense: (batch, neuronas)
            # Revisamos si una neurona es 0 en todas las imágenes
            muertas = np.all(act == 0, axis=0)
            
        total = len(muertas)
        cant_muertas = np.sum(muertas)
        porcentaje = (cant_muertas / total) * 100
        
        print(f"{layer.name:<25} | {total:<8} | {cant_muertas:<8} | {porcentaje:.1f}%")


