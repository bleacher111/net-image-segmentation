# U-Net Image Segmentation (PyTorch)

Proyecto end-to-end de segmentación de imágenes binarias utilizando **U-Net en PyTorch**, con **Dice loss**, **Dice metric**, early stopping, visualización de curvas de entrenamiento y pipeline de inferencia listo para generar predicciones tipo Kaggle.

## Project Overview

El proyecto implementa un pipeline completo de deep learning para segmentación de imágenes, incluyendo:

- Carga y preprocesamiento de datos  
- Definición del modelo U-Net  
- Entrenamiento y validación  
- Implementación de Dice metric y Dice loss  
- Early stopping  
- Visualización de métricas  
- Pipeline de inferencia para generar predicciones  

El flujo principal del proyecto se encuentra dentro de la carpeta notebooks.

## Repository Structure

La estructura del proyecto sigue un formato profesional estándar:

- notebooks → flujo principal de experimentación, entrenamiento y evaluación  
- src → código reutilizable (métricas, funciones auxiliares, helpers de entrenamiento)  
- scripts → carpeta prevista para futuras versiones ejecutables  
- results → outputs como modelos entrenados y predicciones  
- reports → visualizaciones, gráficos y ejemplos cualitativos  
- requirements.txt  
- README.md  

## Model and Methodology

- Arquitectura: **U-Net** (encoder–decoder convolucional)  
- Tarea: **segmentación binaria de imágenes**  
- Función de pérdida: **Soft Dice Loss**  
- Métrica de evaluación: **Dice Coefficient sobre predicciones binarizadas**  
- Optimizador: Adam  
- Estrategias aplicadas:
  - Monitoreo en validación  
  - Early stopping  
  - Curvas de pérdida y métrica  

Las funciones de soporte están implementadas en el archivo src/utils.py.

## Setup

El proyecto está preparado para ejecutarse en un entorno estándar de Python con las dependencias definidas en requirements.txt.  
Puede utilizarse tanto en CPU como en GPU según la configuración del entorno.

## How to Run

Abrir el notebook principal ubicado en la carpeta notebooks y ejecutar las celdas de arriba hacia abajo.

El flujo incluye:

- Carga y preparación de datos  
- Entrenamiento del modelo  
- Evaluación en validación  
- Visualización de métricas  
- Inferencia sobre nuevas imágenes  

El notebook está organizado para que todo el pipeline sea entendible y reproducible.

## Results

Sección editable para agregar resultados reales del experimento.

Formato sugerido:

- Mejor Dice en validación: TBD  
- Epochs utilizados: TBD  
- Batch size: TBD  
- Optimizador: Adam  
- Observaciones:
  - Qué ayudó a mejorar la performance: TBD  
  - Principales limitaciones observadas: TBD  

## Future Improvements

Este proyecto fue diseñado como una base sólida y extensible. Posibles mejoras futuras incluyen:

- Separar la lógica del notebook en scripts ejecutables  
- Incorporar configuración externa  
- Entrenamiento con mixed precision  
- Mejoras en post-procesamiento de máscaras  
- Automatización de búsqueda de hiperparámetros  
- Tests unitarios para métricas y funciones clave  

## Author

Bruno Dinello / Carlos Dutra / Lorenzo Foderé
Machine Learning & Data Science  
LinkedIn: www.linkedin.com/in/bruno-dinello  
GitHub: bleacher111  
