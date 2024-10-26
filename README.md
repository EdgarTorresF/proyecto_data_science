
# Proyecto en Streamlit de un Modelo de Detección de Fraudes usando Redes Neuronales

## Creador: Edgar Torres
## Fecha: 26 de Octubre 2024
## Descripcion: Proyecto en Streamlit
Se replico un modelo de deteccion de fraudes en transacciones de una terminal usando redes neuronales con el Modulo de Keras de Tensorflow
## Problema que se desea resolver
Los fraudes son un problema muy comun que genera grandes perdidas a los comercios diariamente, con este modelo se intenta predecir si las transacciones de una terminal son fraudulentas para poder detenerlas antes de que se genere una perdida al comecio.
## Explicar qué tecnología se usarán
Para la creacion del modelo se usaron distintos programas:
- Google Colab
- Visual Studio Code
- GitHub
- Streamlit
- El programa se creo usando el Modulo de Keras de Tensorflow 
## Objetivos
Crear un modelo que pueda predecir las transacciones fraudulentas de una terminal utilizando un modelo de redes neutonales y adaptarlo para su uso en Streamlit

## Descripción de cómo se obtuvieron los datos.
La base de datos se obtuvo de: https://github.com/Fraud-Detection-Handbook/simulated-data-raw/tree/main/data.

La base de datos esta formada por multiples archivos tipo .pkl por lo que de igual manera se creo un notebook para poder generar un solo archivo y despues poder usarlo en nuestro modelo: https://github.com/EdgarTorresF/proyecto_data_science/blob/main/Merge_Pickle_files_from_Google_Drive.ipynb
## Conclusiones
Se pudo generar un modelo de redes neuronales que puede predecir transacciones fraudulentas usando una base de datos simulada de una terminal bancaria.

El modelo generado tiene un % de True Positives de cerca de un 70% generando menos de 1% de Falsos Positivos.

El modelo se adapto para su visualizacion en Streamlit.
## Referencias
El modelo se replico de: https://deepnote.com/workspace/Deepnote-8b0ebf6d-5672-4a8b-a488-2dd220383dd3/project/Detecting-credit-card-fraud-using-TensorFlow-and-Keras-9848c5e4-f0a5-4c88-987d-d3b1c171d1be/notebook/notebook-6d1e4b10ed2d479baa05499bffd7f956

