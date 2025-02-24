import cv2
import numpy as np

# Calcular IoU
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Calcular el área de intersección
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Calcular el área de cada caja
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Calcular la unión
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# Devuelve la región de intersección entre dos bboxes.
def compute_intersection(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    if xA < xB and yA < yB:
        return (xA, yA, xB, yB)  # Región de intersección
    return None


def fill_rect(mask, box, label):
    """Rellena un rectángulo (box) en la 'mask' con el 'label'."""
    x1, y1, x2, y2 = map(int, box)
    mask[y1:y2, x1:x2] = label

def draw_regions(image, gt_boxes, pred_boxes):
    """Crea una máscara de etiquetas y asigna colores TP, FP, FN."""
    iou_threshold = 0.5
    try:
        h, w, _ = image.shape
    except:
        h, w = image.shape
    
    # Creamos una máscara de UN SOLO canal con 0 por defecto (sin clasificación)
    # Para un problema con 3 categorías (TP, FP, FN), puedes usar, por ejemplo:
    # 1 = TP, 2 = FP, 3 = FN
    label_mask = np.zeros((h, w), dtype=np.uint8)
    
    matched_pred = set()

    for gt in gt_boxes:
        matched = False
        
        for j, pred in enumerate(pred_boxes):
            intersection = compute_intersection(gt, pred)
            iou = compute_iou(gt, pred)
            
            if intersection and iou >= iou_threshold:
                matched = True
                matched_pred.add(j)
                
                # Marcar la INTERSECCIÓN como TP (1)
                fill_rect(label_mask, intersection, 1)
                
                # Marcar la parte de GT que NO está en la intersección como FN (3)
                # (Sección izquierda)
                left_region = (gt[0], gt[1], intersection[0], gt[3])
                if left_region[0] < left_region[2] and left_region[1] < left_region[3]:
                    fill_rect(label_mask, left_region, 3)
                
                # (Sección derecha)
                right_region = (intersection[2], gt[1], gt[2], gt[3])
                if right_region[0] < right_region[2] and right_region[1] < right_region[3]:
                    fill_rect(label_mask, right_region, 3)
                
                # (Sección arriba)
                top_region = (intersection[0], gt[1], intersection[2], intersection[1])
                if top_region[0] < top_region[2] and top_region[1] < top_region[3]:
                    fill_rect(label_mask, top_region, 3)
                
                # (Sección abajo)
                bottom_region = (intersection[0], intersection[3], intersection[2], gt[3])
                if bottom_region[0] < bottom_region[2] and bottom_region[1] < bottom_region[3]:
                    fill_rect(label_mask, bottom_region, 3)
                
                # Marcar la parte de la PRED que NO está en la intersección como FP (2)
                # (Sección izquierda)
                left_region = (pred[0], pred[1], intersection[0], pred[3])
                if left_region[0] < left_region[2] and left_region[1] < left_region[3]:
                    fill_rect(label_mask, left_region, 2)
                
                # (Sección derecha)
                right_region = (intersection[2], pred[1], pred[2], pred[3])
                if right_region[0] < right_region[2] and right_region[1] < right_region[3]:
                    fill_rect(label_mask, right_region, 2)
                
                # (Sección arriba)
                top_region = (intersection[0], pred[1], intersection[2], intersection[1])
                if top_region[0] < top_region[2] and top_region[1] < top_region[3]:
                    fill_rect(label_mask, top_region, 2)
                
                # (Sección abajo)
                bottom_region = (intersection[0], intersection[3], intersection[2], pred[3])
                if bottom_region[0] < bottom_region[2] and bottom_region[1] < bottom_region[3]:
                    fill_rect(label_mask, bottom_region, 2)
                
        # Si no se encontró ninguna predicción que coincida con el GT
        if not matched:
            fill_rect(label_mask, gt, 3)  # FN (3)
    
    # Marcar las preds que no coinciden con ningún GT
    for j, pred in enumerate(pred_boxes):
        if j not in matched_pred:
            fill_rect(label_mask, pred, 2)  # FP (2)
    
    # Ahora mapeamos la label_mask a una máscara de 3 canales con colores
    # 0 = Sin clasificación (transparente)
    # 1 = TP (verde)   → (0, 255, 0)
    # 2 = FP (rojo)    → (255, 0, 0)
    # 3 = FN (azul)    → (0, 0, 255)
    
    color_map = {
        0: (0, 0, 0),        # Sin color
        1: (15, 60, 15),      # Verde oscuro para TP
        2: (0, 0, 60),      # Rojo oscuro para FP
        3: (100, 0, 0),      # Azul oscuro para FN
    }
    
    # Creamos una máscara de 3 canales para la visualización
    color_mask = np.zeros((h, w, 3), dtype=np.float32)
    for label, color in color_map.items():
        color_mask[label_mask == label] = color
    
    # Finalmente, combinamos color_mask con la imagen original
    # alpha controla la transparencia de la superposición
    alpha = 1.0
    image3c = np.stack([image] * 3, axis=-1)
    blended = cv2.addWeighted(image3c, 0.8, color_mask, alpha, 0)

    return blended



# Dibujar GTs y predicciones
#image = draw_regions(image, boxes_gt, boxes)