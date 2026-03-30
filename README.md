[![Repositorio en GitHub](https://img.shields.io/badge/GitHub-Ver_Repositorio-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/alejandrotm22/TextClassification)

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

A continuación detallamos el pipeline completo que hemos seguido a través de los distintos notebooks, centrándonos en el proceso y las herramientas empleadas.

## 1. Preprocesamiento (`01_Preprocessing.ipynb`)
Nuestra primera prioridad ha sido la integridad y riqueza del dato. No nos hemos limitado a una limpieza superficial; hemos diseñado un pipeline de normalización estructural profundo guiado por el motor de **Spacy**:
- Hemos identificado y preservado **patrones de formato** (negritas, estructuras de listas y saltos de línea), convirtiéndolos en **etiquetas semánticas** (TAGS) para que los modelos reconozcan el sesgo de presentación de cada IA.
- Hemos ejecutado una lematización exhaustiva mediante el pipeline de **Spacy** (`en_core_web_sm`), permitiéndonos analizar la raíz léxica y la riqueza gramatical de cada respuesta sin el ruido de la flexión de palabras, además de aislar adecuadamente las Stopwords problemáticas.

## 2. Análisis Exploratorio de Datos / EDA (`02_EDA.ipynb`)
Nos centramos en tratar de identificar posibles "huellas estilísticas" empíricas. Varios descubrimientos clave fundamentan el posterior Feature Engineering:
- **Longitud y Riqueza Léxica (TTR / STTR):** Vimos que *Grok* resultó tener un vocabulario amplísimo (0.60 STTR) frente a la extrema repetición léxica de *Llama-3.2*.
- **Formateo Computacional:** Confirmamos con las métricas que GPT y Gemini abusan bastante del markdown (negritas), Llama de las listas construidas numéricamente, y Mixtral genera textos desnudos o muy planos.
- **N-gramas Discriminatorios (TF-IDF One-vs-Rest):** En vez del clásico TF-IDF global (que genera sesgos de vocabulario por culpa de los tópicos de los textos), usamos esta variante analítica asimétrica para aislar exclusivamente los *marcadores exclusivos de estilo* que usa cada modelo (ej. las contracciones perennes de *Grok*, o fórmulas introductorias puras).
- **Proyección Bidimensional LSA 2D / PCA:** Validación inicial visual de la dispersión de las clases en el crudo espacio multidimensional *Bag of Words*.

### 🛡️ Estrategia de Evaluación: Test *Out-of-Distribution* (OOD)
Un detalle metodológico vital en este proyecto ha sido la forma de dividir nuestro dataset para la evaluación. Dado que los textos generados abordan diversas temáticas, si aplicábamos una división aleatoria tradicional (ej. 80/20), corríamos el riesgo de sufrir fuga de información (*data leakage*): textos con un vocabulario o estructura temática muy similar habrían caído tanto en el conjunto de *Train* como en el de *Test*.

Para evitar que los modelos "hicieran trampa" memorizando el vocabulario de un tema específico, implementamos una estrategia **Out-of-Distribution (OOD)**. Decidimos **aislar un tópico entero del dataset** y reservarlo exclusivamente para el conjunto de *Test*. 

De esta forma, obligamos a los algoritmos a aprender la verdadera **huella estilística y estructural** de cada IA, garantizando que cuando evaluamos su precisión (Accuracy), lo estamos haciendo sobre un contexto temático que el modelo jamás ha visto durante su entrenamiento.

## 3. Machine Learning Baseline (`03_Machine_Learning.ipynb`)
Para establecer un punto de comparación sólido, hemos implementado modelos de aprendizaje supervisado clásico:
- Hemos desarrollado un sistema de vectorización híbrida, combinando TF-IDF a nivel de palabra y n-gramas, logrando capturar tanto el vocabulario preferente como las secuencias sintácticas recurrentes.
- Hemos entrenado y contrapuesto a un **Multinomial Naive Bayes (MNB)** frente a algoritmos de hiperplano separador óptimo.
- **Resultados:** Con el uso de un **Support Vector Machine (Linear SVC),** hemos alcanzado una precisión altísima de prácticamente el **93% en Test**, confirmando que existen divergencias estadísticas significativas en la forma en que cada modelo estructura inicialmente la información.

## 4. Deep Learning desde Cero (`04_Deep_Learning.ipynb`)
Avanzamos hacia el uso de modelos de Deep Learning construidos manual y metódicamente desde cero, más potentes y con más parámetros que los modelos de Machine Learning.
- Hemos implementado arquitecturas **Bi-LSTM (Bidirectional Long Short-Term Memory)** que recogen datos provenientes de capas de **Embeddings**, con el fin de capturar la dependencia temporal y el inmenso contexto bidireccional del texto analizando cómo fluye la narrativa de cada modelo.
- **Manejo del Overfitting:** Durante las primeras iteraciones la red mostró graves picos de memorización espuria (llegando a asentar un 100% de precisión entrenando frente a un 85% de error por ruido en test). Para combatirlo, escalamos la arquitectura reduciendo las dimensiones y metiéndole regularización muy agresiva en el cuello de botella: creamos un *Stacked Bi-LSTM*, un **Dropout masivo intercapa del 50%** y aplicamos penalizaciones de olvido por *Weight Decay (L2)* vía optimizador Adam. Tras esta limpia, la red demostró estabilidad ascendiendo a un contundente 87% generalizando sobre datos no vistos.

## 5. Transformers & Fine-Tuning de BERT (`05_BERT_FineTuning.ipynb`)
Terminamos la parte central de clasificación con un Transfer Learning sobre un Transformer inmenso. Concretamente `bert-base-uncased`, que cuenta con 110M de parámetros de fábrica.
- **Optimización de Hardware:** Hemos entrenado el modelo localmente en una NVIDIA RTX 4070 (8GB VRAM). Para procesar secuencias contiguas de 512 tokens (evitando el truncamiento de información crítica discursiva), hemos configurado e ingeniado técnicas de Precisión Mixta (AMP) y Recorte de Gradientes (Gradient Clipping).
- **Arquitectura y Entrenamiento:** Hemos optado por *congelar (freeze)* deliberadamente las primeras 6 capas del codificador para preservar el sólido conocimiento lingüístico general adquirido, permitiendo que únicamente las capas neuronales superiores se especialicen en exclusividad sobre la detección de nuestros tenues matices estilométricos; optimizando los pesos con `AdamW` y ajustando la tasa de aprendizaje mediente un scheduler lineal con warmup (garantizando así una convergencia estable y preservando el sesgo).
- **Resultados:** El modelo barrió por completo a los algoritmos basados en la simple semántica recurrente, marcando un sólido **93% de test accuracy** y sin aparentes brechas en la curva al validarlo (Train/Val gap estanco por debajo del punto).

## 6. Parte Generativa: Ingeniería Inversa de Prompts (`06_Generative.ipynb`)
Como validación final conceptual, lúdica y clímax de la comprensión lectora del algoritmo, hemos explorado la dirección técnica diametralmente inversa del lenguaje:
- Hemos descargado e instanciado una arquitectura **T5-Small** pre-entrenada por Google acoplando tareas de texto puro a texto (Text-to-Text).
- Tras modificar el vector diana, el objetivo de la red fue intentar acometer con éxito pura *ingeniería inversa* de la intención humana primigenia. A partir de someter al modelo a digerir la respuesta desmedidamente extensa elaborada por una IA cualquiera (anexa al tag prescriptivo `"generate question: "`), este T5 generativo demostró que es empíricamente capaz de deducir, decodificar secuencialmente e hilar por síntesis computacional constructiva la instrucción original originaria (nuestro prompt), validando la irrefutable coherencia semántica implícita que sigue existiendo entre un output inmenso de la IA y el diminuto texto humano.

## 🛠️ Especificaciones Técnicas
Para dotar de soporte matemático y de memoria a las arquitecturas mencionadas, el core se ha erigido sobre los siguientes pilares:
- **Frameworks Principales:** PyTorch, Hugging Face Transformers Custom Pipeline.
- **Procesamiento Lingüístico de Base:** Spacy (`en_core_web_sm`) y un conjunto de herramientas estadísticas con Scikit-learn.
- **Infraestructura de Cómputo Local:** Cargas aceleradas en GPU con arquitectura Ada Lovelace acoplada (NVIDIA RTX 4070, 8GB). La barrera temporal e instruccional de la VRAM se eludió mediante programación avanzada de acumulación por gradientes segmentados y Precisión Computacional Mixta.
## 📊 Análisis Comparativo de Modelos
### Tabla Global de Rendimiento
La siguiente tabla resume el rendimiento global de los modelos frente al conjunto de datos de evaluación (OOD - Out of Distribution | Space Exploration).  

| Modelo | Exactitud (Accuracy) | F1-Score (Macro) | F1-Score (Weighted) | 
| --------- | --------- | --------- | ---- |
| **Linear SVM**    | **0.93**    | 0.93    | **0.93** |
| **BERT**   | 0.93    | **0.94**   | 0.93 |
| **Multinomial Naive Bayes**   | 0.88    | 0.89   | 0.88 |
| **LSTM**   | 0.87    | 0.88   | 0.87 |

### 🔍 Rendimiento por Clase (F1-Score en Test)

Para entender mejor en qué modelos de lenguaje fallan o aciertan nuestros clasificadores, desglosamos el **F1-Score** por cada autor generado. Hemos resaltado en negrita el mejor resultado para cada IA:

| Autor del Texto | Naive Bayes | Linear SVM | LSTM | BERT (Fine-Tuned) |
| :--- | :---: | :---: | :---: | :---: |
| **Grok-3-mini** | 0.93 | 0.96 | 0.92 | **0.97** |
| **Gemini-2.5-lite** | 0.89 | 0.91 | 0.88 | **0.97** |
| **Mixtral-8x7b** | 0.89 | **0.94** | 0.84 | 0.92 |
| **GPT-4.1-nano** | 0.86 | **0.92** | 0.89 | 0.89 |
| **Llama-3.2-1b** | 0.84 | **0.90** | 0.84 | **0.90** |

### 🧠 Conclusiones del Análisis

1. **El Sorprendente Empate (SVM vs. BERT):** El modelo tradicional **Linear SVM** ha logrado empatar en *Accuracy* global (0.93) con **BERT**, un modelo Transformer de última generación. Esto nos indica que las características lineales de los textos (TF-IDF, n-gramas) separan excepcionalmente bien las "muletillas" estructurales de cada LLM. Sin embargo, hay que destacar que SVM alcanzó un `1.00` perfecto en el entrenamiento, lo que indica un fuerte *overfitting* (sobreajuste), aunque demostró ser sorprendentemente robusto al generalizar en el set de Test (Out of Distribution).

2. **BERT domina las clases difíciles:**
   Aunque empatan en exactitud global, BERT demuestra su superioridad contextual al clasificar modelos que a priori son muy parecidos o tienen un estilo más camaleónico. Por ejemplo, BERT eleva la detección de *Gemini-2.5-lite* a un F1-Score de **0.97** (frente al 0.91 de SVM).

3. **El sufrimiento del LSTM con datos OOD:**
   La arquitectura recurrente (LSTM) parecía prometer mucho en entrenamiento (*Accuracy* del 0.97), pero es la que más sufre al enfrentarse a los datos de evaluación (baja a 0.87). Resulta muy revelador ver cómo se desploma su rendimiento (*F1-Score: 0.84*) al intentar clasificar los textos de **Mixtral-8x7b** y **Llama-3.2-1b**. Al observar las métricas detalladas, vemos desequilibrios fuertes: con Mixtral tiene mucha precisión pero poco *recall* (0.74), y con Gemini tiene un *recall* casi perfecto (0.99) pero baja precisión (0.79). Esto demuestra que le cuesta abstraer el estilo general frente a los Transformers.

4. **Naive Bayes como Baseline ideal:**
   Con un 0.88 de *Accuracy* y siendo el modelo más rápido de entrenar (fracciones de segundo), NB se consolida como un excelente punto de partida (*baseline*) para esta tarea.

> **Veredicto Final:** Para un despliegue en producción donde los recursos y tiempos de inferencia sean críticos, **Linear SVM** es la opción ganadora por su impecable relación rendimiento/coste computacional. Sin embargo, si se busca la máxima fiabilidad semántica, evitar el sobreajuste extremo y se dispone de aceleración por GPU, **BERT** es la arquitectura definitiva.

## 👥 Autores del Proyecto

Este proyecto ha sido desarrollado por:

* **Alejandro Torres Martínez** - [GitHub](https://github.com/alejandrotm22) | [LinkedIn](https://www.linkedin.com/in/alejandro-torres-martinez-b65490301/)
* **Alejandro Coman Venceslá** -  [GitHub](https://github.com/AleComan) | [LinkedIn](https://www.linkedin.com/in/aleecv/)

