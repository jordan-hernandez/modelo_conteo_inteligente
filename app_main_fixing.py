import supervision as sv
from ultralytics import YOLO
import numpy as np
import torch
import cv2
import threading
import datetime
from flask import Flask, Response, jsonify
from flask_socketio import SocketIO
from pymongo import MongoClient
from send_data import send_data_to_orion_ld
import time 

import warnings
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")
# -------------------------------
# Configuración y variables globales
# -------------------------------
COCO_CLASS_NAMES = {
    0: "person",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# Cargar el modelo YOLOv5x6 y configurarlo
model = torch.hub.load('ultralytics/yolov5', 'yolov5x6')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.conf = 0.5  # Umbral global de confianza

# Inicializar el tracker (ByteTrack)
tracker = sv.ByteTrack()

# Fuente de video (puede ser RTSP o video local)
video_path = "rtsp://admin:CentroCali@172.16.30.51:554/onvif1"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: No se pudo abrir el stream de video.")
    exit()

# Definir zonas (polígonos) para una resolución 1920x1080
persons_polygon = np.array([[1230,1073],
                             [ 809,837],
                             [1248,817]], np.int32)
vehicles_polygon = np.array([
    [400,900],
    [1520,900],
    [1920,1080],
    [1520,1080],
    [400,1080],
    [0,1080]
], np.int32)

zones = [
    sv.PolygonZone(polygon=persons_polygon),
    sv.PolygonZone(polygon=vehicles_polygon)
]

colors = sv.ColorPalette.DEFAULT
zone_annotators = [
    sv.PolygonZoneAnnotator(
        zone=zone,
        color=colors.by_idx(idx),
        thickness=6,
        text_scale=4
    )
    for idx, zone in enumerate(zones)
]

box_annotator = sv.BoundingBoxAnnotator(thickness=4)

# Clases permitidas por zona
allowed_classes = {
    0: [0],
    1: [2, 3, 5, 7]
}

# Diccionarios para conteo y registro de tracker_ids
zone_counts = {
    0: {cls: 0 for cls in allowed_classes[0]},
    1: {cls: 0 for cls in allowed_classes[1]}
}
zone_counted_tracker = {0: set(), 1: set()}

# Lock para sincronizar acceso a los contadores
data_lock = threading.Lock()

# Variable global para evitar duplicados
last_sent_counts = None

# Intervalo para reset (en segundos); ajústalo según lo deseado
RESET_INTERVAL = 90

# -------------------------------
# Configuración de Flask, SocketIO y MongoDB
# -------------------------------
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Usar cadena de conexión completa para MongoDB
MONGO_URI = "mongodb://localhost:27017"
client = MongoClient(MONGO_URI)
db = client["Conteo_multiple"]
collection = db["Conteo"]

# -------------------------------
# Función para procesar un frame
# -------------------------------
def process_frame(frame):
    global zone_counts, zone_counted_tracker
    results = model(frame, size=1280)
    detections = sv.Detections.from_yolov5(results)
    detections = detections[detections.confidence > 0.5]
    if len(detections) > 0:
        detections = tracker.update_with_detections(detections)
        # Procesar cada zona
        for zone_idx, (zone, zone_annotator) in enumerate(zip(zones, zone_annotators)):
            mask_zone = zone.trigger(detections=detections)
            zone_detections = detections[mask_zone]
            allowed = allowed_classes[zone_idx]
            mask_allowed = np.isin(zone_detections.class_id, allowed)
            zone_detections = zone_detections[mask_allowed]
            # Control especial para trucks en zona de vehículos
            if zone_idx == 1:
                truck_threshold = 0.75
                truck_mask = (zone_detections.class_id == 7) & (zone_detections.confidence >= truck_threshold)
                other_mask = (zone_detections.class_id != 7)
                final_mask = truck_mask | other_mask
                zone_detections = zone_detections[final_mask]
            # Etiquetar detecciones
            labels = []
            for i in range(len(zone_detections.xyxy)):
                tid = zone_detections.tracker_id[i]
                cls_id = int(zone_detections.class_id[i])
                labels.append(f"ID:{tid} {COCO_CLASS_NAMES[cls_id]}")
            zone_detections.label = labels
            frame = box_annotator.annotate(scene=frame, detections=zone_detections)
            # Actualizar conteos evitando duplicados
            for i in range(len(zone_detections.xyxy)):
                tid = zone_detections.tracker_id[i]
                cls_id = int(zone_detections.class_id[i])
                if tid not in zone_counted_tracker[zone_idx]:
                    zone_counts[zone_idx][cls_id] += 1
                    zone_counted_tracker[zone_idx].add(tid)
            frame = zone_annotator.annotate(scene=frame)
            color_bgr = colors.by_idx(zone_idx).as_bgr()
            count_texts = [f"{COCO_CLASS_NAMES[cls_id]}: {zone_counts[zone_idx][cls_id]}" for cls_id in allowed]
            text_to_show = " | ".join(count_texts)
            cv2.putText(frame, text_to_show, tuple(zone.polygon[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color_bgr, 4)
    return frame

# -------------------------------
# Generador para streaming de video
# -------------------------------
def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = process_frame(frame)
        ret2, buffer = cv2.imencode('.jpg', frame)
        if not ret2:
            continue
        socketio.emit("frame", buffer.tobytes())
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# -------------------------------
# Rutas de Flask
# -------------------------------
@app.route("/")
def index():
    return "Streaming en tiempo real con conteo, JSON y base de datos."

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/conteo")
def get_conteo():
    with data_lock:
        conteo = {
            "personas": zone_counts[0],
            "vehiculos": zone_counts[1]
        }
    return jsonify(conteo)

# -------------------------------
# Función para guardar datos en MongoDB y reiniciar conteo
# -------------------------------
def reset_counts():
    global zone_counts, zone_counted_tracker, last_sent_counts

    # Tomar snapshot atómico y reiniciar contadores dentro del lock
    with data_lock:
        timestamp = datetime.datetime.now().isoformat()
        current_counts = {
            "personas": {COCO_CLASS_NAMES[int(k)]: v for k, v in zone_counts[0].items()},
            "vehiculos": {COCO_CLASS_NAMES[int(k)]: v for k, v in zone_counts[1].items()}
        }
        total_count = sum(current_counts["personas"].values()) + sum(current_counts["vehiculos"].values())

        # Reiniciar contadores inmediatamente para minimizar bloqueo
        zone_counts[0] = {cls: 0 for cls in allowed_classes[0]}
        zone_counts[1] = {cls: 0 for cls in allowed_classes[1]}
        zone_counted_tracker[0] = set()
        zone_counted_tracker[1] = set()

    # Si no hubo detección, no enviar datos
    if total_count == 0:
        print("Advertencia: Conteo cero detectado, omitiendo guardado.")
    else:
        # Evitar duplicados comparando con el último snapshot enviado (excluyendo timestamp)
        if last_sent_counts == current_counts:
            print("Snapshot duplicado, omitiendo guardado.")
        else:
            last_sent_counts = current_counts
            conteo_actual = {
                "personas": current_counts["personas"],
                "vehiculos": current_counts["vehiculos"],
                "timestamp": timestamp
            }
            # Guardar en MongoDB
            try:
                result = collection.insert_one(conteo_actual)
                if result.acknowledged:
                    print(f"Inserción exitosa en MongoDB. ID: {result.inserted_id}")
                else:
                    print("Error: MongoDB no confirmó la escritura")
            except Exception as mongo_error:
                print(f"Error específico de MongoDB: {mongo_error}")
                print(f"URI: {MONGO_URI} | DB: {db.name} | Colección: {collection.name}")

            # Enviar datos a Orion-LD de forma asíncrona para no bloquear
            def send_orion(data):
                try:
                    send_data_to_orion_ld(**data)
                except Exception as e:
                    print(f"Error al enviar datos a Orion-LD: {str(e)}")
            for category in ["personas", "vehiculos"]:
                for obj_type, count in current_counts[category].items():
                    send_data = {
                        "category": category,
                        "obj_type": obj_type,
                        "count": count,
                        "timestamp": timestamp
                    }
                    threading.Thread(target=send_orion, args=(send_data,)).start()

    # Reprogramar la próxima ejecución después del intervalo definido
    threading.Timer(RESET_INTERVAL, reset_counts).start()

# -------------------------------
# Inicialización de la aplicación
# -------------------------------
if __name__ == "__main__":
    try:
        client.admin.command("ping")
        print("Conexión exitosa a MongoDB")
    except Exception as e:
        print("Error al conectar a MongoDB:", e)
        exit(1)
    reset_counts()  # Inicia el ciclo de guardado y reseteo
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
