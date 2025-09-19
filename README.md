EXTRACTOR DE ETIQUETAS NER

El codigo se encarga de extraer ciertas palabras de un texto que pertenezcan a las etiquetas de 'PERSONA', 'UBICACION', 'ORGANIZACION' y 'MISCELANEOS'. se utiliza el modelo preentrenado 'bert-spanish-cased-finetuned-ner' para realizar esta tarea. En el prompt se ingresa un texto, y en la salida se entregan las palabras que se extrajeron con su respectiva etiqueta.

El archivo 'requirements.txt' contiene las librerias necesarias para su funcionamiento. pip install -r requirements.txt