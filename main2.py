import supervision as sv
from ultralytics import YOLO
import numpy as np
import torch
import cv2

# Mapeo de nombres de clases COCO
COCO_CLASS_NAMES = {
    0: "person",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# Cargar el modelo YOLOv5x6
model = torch.hub.load('ultralytics/yolov5', 'yolov5x6')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.conf = 0.5  # Umbral global de confianza

# Inicializar el tracker ByteTrack con parámetros personalizados
tracker = sv.ByteTrack(
    track_activation_threshold=0.25,  # Umbral de confianza para activar una pista
    lost_track_buffer=30,             # Número de fotogramas en el búfer para pistas perdidas
    minimum_matching_threshold=0.8,   # Umbral para emparejar pistas con detecciones
    frame_rate=15                     # Tasa de fotogramas del video
)

# Fuente de video (RTSP en este ejemplo)
video_path = "rtsp://admin:CentroCali@172.16.30.55:554/onvif1"
#video_path= 0
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: No se pudo abrir el stream de video.")
    exit()

# Obtener propiedades del video para configurar el VideoWriter
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Definir el codec y crear el objeto VideoWriter para guardar el output
output_path = 'resultado.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Definir polígonos (zonas)
# Zona de personas (parte superior)
persons_polygon = np.array([[828, 672],
                             [1191, 912],
                             [1431, 779],
                             [1146, 606]], np.int32)
# Zona de vehículos (parte inferior)
persons_polygon= np.array([[1230,1073],
 [ 809 , 837],
 [1248  ,817]], np.int32 ) 
vehicles_polygon = np.array([[400, 900],
                             [1520, 900],
                             [1920, 1080],
                             [1520, 1080],
                             [400, 1080],
                             [0, 1080]], np.int32)

# Crear las zonas
zones = [
    sv.PolygonZone(polygon=persons_polygon),
    sv.PolygonZone(polygon=vehicles_polygon)
]

# Crear anotadores para las zonas
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

# Anotador para las bounding boxes
box_annotator = sv.BoundingBoxAnnotator(thickness=4)

# Definir las clases permitidas para cada zona:
# Zona de personas: solo clase 0 ("person")
# Zona de vehículos: se consideran clases 2 ("car"), 3 ("motorcycle"), 5 ("bus") y 7 ("truck")
allowed_classes = {
    0: [0],
    1: [2, 3, 5, 7]
}

# Inicializar contadores por zona (por clase)
zone_counts = {
    0: {cls: 0 for cls in allowed_classes[0]},
    1: {cls: 0 for cls in allowed_classes[1]}
}
# Registro de tracker_ids ya contados (por zona)
zone_counted_tracker = {
    0: {},  # key: tracker_id, value: clase asignada (fija)
    1: {}
}

# (Opcional) Diccionario para fijar la etiqueta asignada la primera vez que se detecta un objeto
zone_fixed_labels = {
    0: {},  # para zona de personas
    1: {}   # para zona de vehículos
}

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo leer el frame.")
        break

    # Realizar la detección con YOLO
    results = model(frame, size=1280)
    detections = sv.Detections.from_yolov5(results)
    detections = detections[detections.confidence > 0.5]

    if len(detections) > 0:
        # Actualizar el tracker para asignar tracker_id a cada detección
        detections = tracker.update_with_detections(detections)

        # Procesar cada zona
        for zone_idx, (zone, zone_annotator) in enumerate(zip(zones, zone_annotators)):
            # Filtrar detecciones dentro del polígono de la zona
            mask_zone = zone.trigger(detections=detections)
            zone_detections = detections[mask_zone]

            # Filtrar detecciones según las clases permitidas para la zona
            allowed = allowed_classes[zone_idx]
            mask_allowed = np.isin(zone_detections.class_id, allowed)
            zone_detections = zone_detections[mask_allowed]

            # Si es la zona de vehículos, aplicar un umbral mayor para trucks (clase 7)
            # Esto se hace porque los trucks se parecen a los carros, entonces para estar seguros que se tiene un truck se aumenta
            # el umbral de confianza
            if zone_idx == 1:
                truck_threshold = 0.75  # umbral mayor para trucks
                truck_mask = (zone_detections.class_id == 7) & (zone_detections.confidence >= truck_threshold)
                other_mask = (zone_detections.class_id != 7)
                final_mask = truck_mask | other_mask
                zone_detections = zone_detections[final_mask]

            # Crear etiquetas que muestren tracker id y el tipo de objeto, fijando la etiqueta la primera vez
            labels = []
            for i in range(len(zone_detections.xyxy)):
                tid = zone_detections.tracker_id[i]
                current_cls = int(zone_detections.class_id[i])
                if tid in zone_fixed_labels[zone_idx]:
                    fixed_cls = zone_fixed_labels[zone_idx][tid]
                else:
                    fixed_cls = current_cls
                    zone_fixed_labels[zone_idx][tid] = fixed_cls
                label_text = f"ID:{tid} {COCO_CLASS_NAMES[fixed_cls]}"
                labels.append(label_text)
            zone_detections.label = labels

            # Dibujar las bounding boxes en la zona
            frame = box_annotator.annotate(scene=frame, detections=zone_detections)

            # Actualizar el conteo: solo se cuenta si el tracker_id aún no fue contado en la zona
            for i in range(len(zone_detections.xyxy)):
                tid = zone_detections.tracker_id[i]
                fixed_cls = zone_fixed_labels[zone_idx][tid]
                if tid not in zone_counted_tracker[zone_idx]:
                    zone_counts[zone_idx][fixed_cls] += 1
                    zone_counted_tracker[zone_idx][tid] = fixed_cls

            # Anotar el polígono de la zona y mostrar el conteo
            frame = zone_annotator.annotate(scene=frame)
            color_bgr = colors.by_idx(zone_idx).as_bgr()
            count_texts = []
            for cls_id in allowed:
                count_texts.append(f"{COCO_CLASS_NAMES[cls_id]}: {zone_counts[zone_idx][cls_id]}")
            text_to_show = " | ".join(count_texts)
            cv2.putText(
                frame,
                text_to_show,
                tuple(zone.polygon[0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                color_bgr,
                4
            )
    out.write(frame)
    cv2.imshow("Stream", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
