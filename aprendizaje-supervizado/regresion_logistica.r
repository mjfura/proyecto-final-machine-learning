# Instalación de librerías
install.packages(c("text2vec", "glmnet", "Matrix", "caret", "dplyr", "readr"))

# librerías necesarias
library(text2vec)
library(glmnet)
library(Matrix)
library(caret)
library(dplyr)
library(readr)

# Cargar el archivo CSV de entrenamiento
# Modificar la ruta donde dejen el archivo <----OJO
ruta_entrenamiento <- "C:/00-Programacion/R_Proyecto_Final_Supervizado/train.csv"
datos_entrenamiento <- read.csv(ruta_entrenamiento, stringsAsFactors = FALSE, encoding = "UTF-8", header = TRUE)

# Verificar el contenido del dataset
head(datos_entrenamiento)


limpiar_texto <- function(texto) {
    texto <- tolower(texto)
    texto <- removePunctuation(texto)
    texto <- removeNumbers(texto)
    texto <- removeWords(texto, stopwords("es"))
    texto <- stripWhitespace(texto)
    return(texto)
}

# Aplicar la limpieza al texto de los tweets
datos_entrenamiento$text <- sapply(datos_entrenamiento$text, limpiar_texto)

# Verificar los textos limpiados
head(datos_entrenamiento$text)

# Crear un iterador para los textos
iterador_textos <- itoken(datos_entrenamiento$text, preprocessor = tolower, tokenizer = word_tokenizer, progressbar = FALSE)

# Crear un vocabulario y una matriz de términos del documento (DTM)
vocabulario <- create_vocabulary(iterador_textos)
vectorizador <- vocab_vectorizer(vocabulario)
dtm <- create_dtm(iterador_textos, vectorizador)

# Convertir la DTM en una matriz dispersa
dtm_dispersa <- as(dtm, "sparseMatrix")

# Dividir los datos en características (X) y etiquetas (y)
X <- dtm_dispersa
y <- factor(datos_entrenamiento$sentiment, levels = c("negative", "neutral", "positive"))

# Entrenar el modelo de regresión logística multiclase
modelo <- cv.glmnet(X, y, family = "multinomial", type.measure = "class", parallel = TRUE)

# Resumen del modelo
print(modelo)

# Dividir el dataset en entrenamiento y prueba (80%-20%)
set.seed(123) # para reproducibilidad
indice_entrenamiento <- createDataPartition(datos_entrenamiento$sentiment,
    p = .8,
    list = FALSE,
    times = 1
)
datos_entrenamiento_div <- datos_entrenamiento[indice_entrenamiento, ]
datos_prueba <- datos_entrenamiento[-indice_entrenamiento, ]

# Preprocesar el texto del conjunto de prueba
datos_prueba$text <- sapply(datos_prueba$text, limpiar_texto)

# Crear un iterador para los textos de prueba
iterador_textos_prueba <- itoken(datos_prueba$text, preprocessor = tolower, tokenizer = word_tokenizer, progressbar = FALSE)

# Crear la DTM para el conjunto de prueba
dtm_prueba <- create_dtm(iterador_textos_prueba, vectorizador)

# Convertir la DTM en una matriz dispersa
dtm_prueba_dispersa <- as(dtm_prueba, "sparseMatrix")

# Alinear las columnas de dtm_prueba_dispersa con dtm_dispersa
dtm_prueba_dispersa <- dtm_prueba_dispersa[, colnames(dtm_dispersa), drop = FALSE]

# Predecir en el conjunto de prueba
predicciones_prueba <- predict(modelo, dtm_prueba_dispersa, type = "class")

# Crear una matriz de confusión para evaluar el modelo
matriz_confusion <- confusionMatrix(
    factor(predicciones_prueba, levels = c("negative", "neutral", "positive")),
    factor(datos_prueba$sentiment, levels = c("negative", "neutral", "positive"))
)
print(matriz_confusion)

# Cargar el nuevo dataset no clasificado
# MODIFICAR ESTA RUTA para que funcion <-------OJO ACÁ
ruta_nuevos_datos <- "C:/00-Programacion/R_Proyecto_Final_Supervizado/test.csv"
nuevos_datos <- read.csv(ruta_nuevos_datos, stringsAsFactors = FALSE, encoding = "UTF-8", header = TRUE)

# Limpiar el texto de los nuevos tweets
nuevos_datos$text <- sapply(nuevos_datos$text, limpiar_texto)

# Crear un iterador para los textos no clasificados
iterador_nuevos_textos <- itoken(nuevos_datos$text, preprocessor = tolower, tokenizer = word_tokenizer, progressbar = FALSE)

# Crear la DTM para los nuevos tweets
dtm_nuevos <- create_dtm(iterador_nuevos_textos, vectorizador)

# Convertir la DTM en una matriz dispersa
dtm_nuevos_dispersa <- as(dtm_nuevos, "sparseMatrix")

# Alinear las columnas de dtm_nuevos_dispersa con dtm_dispersa
dtm_nuevos_dispersa <- dtm_nuevos_dispersa[, colnames(dtm_dispersa), drop = FALSE]

# Predecir las etiquetas para los nuevos tweets
predicciones_nuevas <- predict(modelo, dtm_nuevos_dispersa, type = "class")

# Añadir las predicciones al dataframe original
nuevos_datos$prediccion_sentimiento <- factor(predicciones_nuevas, levels = c("negative", "neutral", "positive"))

# Guardar los resultados en un nuevo archivo CSV
ruta_salida <- "C:/00-Programacion/R_Proyecto_Final_Supervizado/tweets_clasificados_2.csv"
write.csv(nuevos_datos, ruta_salida, row.names = FALSE)

# Verificar los resultados
head(nuevos_datos)



# Validación del rendimiento del modelo
# Calcular la precisión en el conjunto de prueba
precision <- sum(predicciones_prueba == datos_prueba$sentiment) / length(predicciones_prueba)
cat("Precisión del modelo en el conjunto de prueba: ", precision * 100, "%\n")

# Calcular la precisión en el conjunto de entrenamiento
predicciones_entrenamiento <- predict(modelo, X, type = "class")
precision_entrenamiento <- sum(predicciones_entrenamiento == datos_entrenamiento$sentiment) / length(predicciones_entrenamiento)
cat("Precisión del modelo en el conjunto de entrenamiento: ", precision_entrenamiento * 100, "%\n")
