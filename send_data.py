import requests
import json

# URL del endpoint de Orion-LD
ORION_LD_URL = "http://localhost:1026/ngsi-ld/v1/entities"

def send_data_to_orion_ld(category, obj_type, count, timestamp):
    """
    Envía datos en formato NGSI-LD a Orion-LD.
    
    Parámetros:
    - category: Categoría de la entidad (ej. "personas", "vehiculos").
    - obj_type: Tipo de objeto (ej. "person", "car").
    - count: Valor del conteo (ej. 5, 10).
    - timestamp: Marca de tiempo en formato ISO 8601 (ej. "2025-02-19T12:00:00Z").
    """
    # Construir la entidad NGSI-LD
    data = {
        "id": f"urn:ngsi-ld:{category}:{obj_type}",
        "type": "Conteo",
        "count": {
            "type": "Property",
            "value": count
        },
        "timestamp": {
            "type": "Property",
            "value": timestamp
        },
        "@context": [
            "https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld"
        ]
    }

    # Encabezados HTTP
    headers = {
        "Content-Type": "application/ld+json"
    }

    # Enviar solicitud POST a Orion-LD
    response = requests.post(ORION_LD_URL, data=json.dumps(data), headers=headers)
    if response.status_code == 201:
        print(f"Datos enviados correctamente para {category}:{obj_type}")
    else:
        print(f"Error al enviar datos para {category}:{obj_type}:", response.text)

""" # Ejemplo de uso
conteo_actual = {
    "personas": {
        "person": 5
    },
    "vehiculos": {
        "car": 10,
        "motorcycle": 3,
        "bus": 2,
        "truck": 1
    },
    "timestamp": "2025-02-19T12:00:00Z"
}

# Iterar sobre las categorías y enviar los datos a Orion-LD
for category, counts in conteo_actual.items():
    if category != "timestamp":
        for obj_type, count in counts.items():
            send_data_to_orion_ld(
                category=category,
                obj_type=obj_type,
                count=count,
                timestamp=conteo_actual["timestamp"]
            )  """