from transformers import pipeline

# Cargar el pipeline de "ner" (Named Entity Recognition) con el modelo en español
print("Cargando el modelo NER...")
ner_pipeline = pipeline(
    "ner",
    model="mrm8488/bert-spanish-cased-finetuned-ner",
    grouped_entities=True # Agrupa partes de una entidad
)
print("¡Modelo cargado!")

# El texto que quieres analizar
# texto = (
#     "El CEO de Tesla, Elon Musk, anunció una nueva inversión para SpaceX en el estado de Nuevo León, México." 
#     "La noticia fue confirmada por el presidente Andrés Manuel López Obrador durante su conferencia en"
#     "la Ciudad de México el pasado mes de marzo. Se espera que esta colaboración con el Gobierno de México"
#     "genere miles de empleos en la región norte del país."
# )

texto = (
    "La Dra. Elena García, una reconocida científica del Instituto Cervantes, presentó los resultados de" 
    "su última investigación en una conferencia internacional celebrada en Kioto, Japón. Su estudio," 
    "financiado en parte por la Unión Europea, analiza el impacto de la tecnología desarrollada por la"
    "empresa alemana Siemens en las economías emergentes de Sudamérica. Al evento asistieron personalidades"
    "como el economista jefe del Banco Mundial, David Malpass, y varios representantes de la ONU."
    "La investigación, que comenzó en el verano de 2023, utilizó datos satelitales proporcionados por la"
    "NASA para monitorear los cambios en la región andina. Los hallazgos, publicados en la revista científica"
    "'Nature', sugieren una nueva era de desarrollo industrial para países como Perú y Colombia." 
)

# Extraer las entidades del texto
print("\nAnalizando el texto para encontrar entidades...")
entidades = ner_pipeline(texto)

# Imprimir los resultados de una forma clara
print("\n--- ENTIDADES ENCONTRADAS ---\n")
for entidad in entidades:
    print(f"Texto: '{entidad['word']}'")
    print(f"Categoría: {entidad['entity_group']} (Confianza: {entidad['score']:.2f})\n")