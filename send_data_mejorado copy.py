import requests
import json

# URL del endpoint de Orion-LD
ORION_LD_URL = "http://localhost:1026/ngsi-ld/v1/entities"
def send_data_to_orion_ld(category, obj_type, count, timestamp):
    """
    Envía datos al contexto de Orion-LD.
    Si la entidad no existe, se crea con POST.
    Si la entidad ya existe, se actualiza con PATCH.
    """
    entity_id = f"{category}:{obj_type}"
    entity_uri = f"urn:ngsi-ld:{entity_id}"
    data = {
        "id": entity_uri,
        "type": category,
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
    headers = {
        "Content-Type": "application/ld+json"
    }

    # Intenta crear la entidad con POST
    response = requests.post(ORION_LD_URL, data=json.dumps(data), headers=headers)
    if response.status_code == 201:
        print(f"Entidad creada correctamente para {entity_id}")
    elif response.status_code == 409:  # Entidad ya existe
        # Actualiza la entidad con PATCH
        update_url = f"{ORION_LD_URL}/{entity_uri}/attrs"
        update_data = {
            "count": {
                "type": "Property",
                "value": count
            },
            "timestamp": {
                "type": "Property",
                "value": timestamp
            }
        }
        update_headers = {
            "Content-Type": "application/json"
        }
        update_response = requests.patch(update_url, data=json.dumps(update_data), headers=update_headers)
        if update_response.status_code == 204:
            print(f"Entidad actualizada correctamente para {entity_id}")
        else:
            print(f"Error al actualizar entidad para {entity_id}:", update_response.text)
    else:
        print(f"Error al crear entidad para {entity_id}:", response.text)