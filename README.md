# Clasificación de Prompts Generados por IA 🤖📝
Este repositorio contiene el código y los experimentos desarrollados para un proyecto de Procesamiento de Lenguaje Natural (NLP). El objetivo principal es construir un sistema capaz de clasificar diferentes prompts generados por Inteligencia Artificial, categorizándolos según la IA que los ha generado y la temática específica que abordan.

Para evaluar el mejor enfoque, el proyecto sigue una progresión metodológica que va desde modelos clásicos de Machine Learning hasta el ajuste fino (Fine-Tuning) de modelos Transformer de última generación.

# 📋 Fases del Proyecto
## 1. Preprocesado y Tratamiento de Datos
Antes de entrenar los modelos, el texto crudo pasa por un pipeline de procesamiento lingüístico riguroso utilizando NLTK:

Tokenización: Separación de palabras y signos de puntuación mediante PunktWordTokenizer.

POS Tagging: Etiquetado de categorías gramaticales usando Perceptron Tagger.

Limpieza: Eliminación de Stopwords para reducir el ruido.

Vectorización Híbrida: Extracción de características combinando matrices TF-IDF a nivel de palabra y TF-IDF a nivel de n-gramas, concatenando ambas representaciones para alimentar los modelos clásicos.

## 2. Machine Learning Baseline
Establecemos un punto de referencia (baseline) robusto utilizando técnicas tradicionales de clasificación:

Modelos: Naive Bayes y Support Vector Machines (SVM) con kernel lineal.

Reducción de Dimensionalidad: Aplicación de técnicas como PCA o t-SNE para manejar la alta dimensionalidad generada por la matriz TF-IDF combinada y facilitar la visualización de los datos.

## 3. Deep Learning
Escalamos la complejidad utilizando redes neuronales recurrentes para capturar el contexto secuencial de los prompts:

Embeddings: Capa de entrada inicial utilizando representaciones vectoriales densas (preentrenadas con Word2Vec o GloVe).

Procesamiento Secuencial: Uso de una o dos capas LSTM (específicamente Bi-LSTM) para entender el contexto bidireccional del texto.

Clasificación: Capa densa (Dense) de salida utilizando la función de pérdida Categorical Cross-Entropy, adecuada para nuestra tarea de clasificación multiclase.

## 4. Transformers & Fine-Tuning
Exploramos el estado del arte en NLP utilizando la arquitectura Transformer:

Modelos Base: Uso de modelos eficientes y ligeros como DistilBERT o RoBERTa-base.

Fine-Tuning: Ajuste fino de estos modelos preentrenados sobre nuestro dataset específico para maximizar la precisión en la identificación de la IA generadora y la temática.
