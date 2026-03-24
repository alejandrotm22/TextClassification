# Clasificación de Prompts Generados por IA 🤖📝

Este repositorio contiene una investigación avanzada de Procesamiento de Lenguaje Natural (NLP) centrada en la "huella digital" de los modelos de lenguaje. El objetivo central ha sido determinar si los modelos de lenguaje actuales poseen rasgos distintivos únicos —una "huella dactilar" algorítmica— que permitan identificar su origen basándose exclusivamente en la estructura y el estilo del texto generado.

En este proyecto, se evoluciona desde el análisis estadístico descriptivo hasta el fine-tuning de arquitecturas Transformer, optimizando el rendimiento para entornos de hardware con restricciones de memoria.

# 🎯 Modelos Objetivo

Este proyecto clasifica textos entre los siguientes cinco modelos:

- **x-ai/grok-3-mini**
- **openai/gpt-4.1-nano**
- **google/gemini-2.5-flash-lite**
- **meta-llama/llama-3.2-1b-instruct**
- **mistralai/mixtral-8x7b-instruct**

# 📋 Fases del Proyecto
## 1. Preprocesado y Tratamiento de Datos

Nuestra primera prioridad ha sido la integridad y riqueza del dato. No nos hemos limitado a una limpieza superficial; hemos diseñado un pipeline de normalización estructural:
- Hemos identificado y preservado **patrones de formato** (negritas, estructuras de listas y saltos de línea), convirtiéndolos en **etiquetas semánticas** (TAGS) para que los modelos reconozcan el sesgo de presentación de cada IA.
- Hemos ejecutado una lematización exhaustiva mediante el pipeline de **Spacy**, permitiéndonos analizar la raíz léxica y la riqueza gramatical de cada respuesta sin el ruido de la flexión de palabras.

## 2. Machine Learning Baseline
Para establecer un punto de comparación sólido, hemos implementado modelos de aprendizaje supervisado clásico:
- Hemos desarrollado un sistema de vectorización híbrida, combinando TF-IDF a nivel de palabra y n-gramas, logrando capturar tanto el vocabulario preferente como las secuencias sintácticas recurrentes.
- Resultados: Con el uso de **Linear SVC,** hemos alcanzado una **precisión del 93%**, confirmando que existen divergencias estadísticas significativas en la forma en que cada modelo estructura la información.

## 3. Deep Learning
Avanzamos hacia el uso de modelos de Deep Learning, más potentes y con más parámetros que los modelos de Machine Learning.
- Hemos implementado arquitecturas **Bi-LSTM (Bidirectional Long Short-Term Memory)** con el fin de capturar la dependencia temporal y el contexto bidireccional del texto, analizando cómo fluye la narrativa de cada modelo.

## 4. Transformers & Fine-Tuning
Terminamos la parte de clasificación con un Fine Tuning de un Transformer. Concretamente BERT-Base-Uncached, que cuenta con 110M de parámetros.
- **Optimización de Hardware:** Hemos entrenado el modelo en una NVIDIA RTX 4070 (8GB VRAM). Para procesar secuencias de 512 tokens (evitando el truncamiento de información crítica), hemos configurado técnicas de Precisión Mixta (AMP) y Acumulación de Gradientes.
- **Arquitectura:** Hemos optado por congelar las primeras 6 capas del codificador para preservar el conocimiento lingüístico general, permitiendo que las capas superiores se especialicen exclusivamente en la detección de matices estilométricos.

## 5. Parte Generativa: ¿Qué le preguntamos?
Como validación final de la comprensión del modelo, hemos explorado la dirección inversa del lenguaje:
- Hemos entrenado un modelo **T5-Small** (Text-to-Text) con el objetivo de realizar ingeniería inversa de prompts. A partir de una respuesta extensa, el modelo ha sido capaz de deducir y sintetizar la instrucción original que la generó, validando la coherencia semántica entre el output de la IA y la intención humana.

# 🛠️ Especificaciones Técnicas
- **Frameworks Principales:** PyTorch, Hugging Face Transformers.
- **Procesamiento Lingüístico:** Spacy (en_core_web_sm), Scikit-learn.
- **Infraestructura de Cómputo:** GPU con arquitectura Ada Lovelace y optimización de memoria mediante gradientes acumulados.

# 📊 Tabla comparativa de las técnicas empleadas para la clasificación texto -> modelo
