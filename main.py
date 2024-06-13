import numpy as np
import cv2

def apply_yolo_object_detection(image_to_process):
    height, width, depth = image_to_process.shape
    blob = cv2.dnn.blobFromImage(image_to_process, 1/255, (608, 608), (0, 0, 0), True, False)
    net.setInput(blob)
    outs = net.forward(out_layers)
    class_indexes, class_scores, boxes = ([] for i in range(3))
    objects_count = 0


    for out in outs:
        for obj in out:
            scores = obj[5:]
            class_index = np.argmax(scores)
            class_score = scores[class_index]
            if class_score > 0:
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                obj_width = int(obj[2] * width)
                obj_height = int(obj[3] * height)
                box = [center_x - obj_width // 2, center_y - obj_height // 2, obj_width, obj_height]
                boxes.append(box)
                class_indexes.append(class_index)
                class_scores.append(float(class_score))


    chosen_boxes = cv2.dnn.NMSBoxes(boxes, class_scores, 0.0, 0.4)
    for box_index in chosen_boxes:
        # box_index = box_index[0]
        box_index = box_index
        box = boxes[box_index]
        class_index = class_indexes[box_index]


        if classes[class_index] in classes_to_look_for:
            objects_count += 1
            image_to_process = draw_object_boundling_box(image_to_process, class_index, box)


    final_image = draw_object_count(image_to_process, objects_count)
    return final_image


def draw_object_boundling_box(image_to_process, index, box):
    x, y, w, h = box
    start = (x, y)
    end = (x + w, y + h)
    color = (0, 255, 0)
    width = 2
    final_image = cv2.rectangle(image_to_process, start, end, color, width)

    start = (x, y - 10)
    font_size = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    width = 2
    text = classes[index]
    final_image = cv2.putText(final_image, text, start, font, font_size, color, width, cv2.LINE_AA)

    return final_image


def draw_object_count(image_to_process, objects_count):
    start = (45, 70)
    font_size = 1.5
    font = cv2.FONT_HERSHEY_SIMPLEX
    width = 3
    text = 'objects found: ' + str(objects_count)
    white_color = (255, 255, 255)
    black_outline_color = (0, 0, 0)
    final_image = cv2.putText(image_to_process, text, start, font, font_size, black_outline_color, width * 3, cv2.LINE_AA)
    final_image = cv2.putText(final_image, text, start, font, font_size, white_color, width, cv2.LINE_AA)

    return final_image



### For image handling ####
###########################

# def start_image_object_detection():
#     try:
#         image = cv2.imread("assets/img2.png")
#         image = apply_yolo_object_detection(image)

#         cv2.imshow("Image", image)
#         if cv2.waitKey(0):
#             cv2.destroyAllWindows()
#     except KeyboardInterrupt:
#         pass


# if __name__ == '__main__':
#     # net = cv2.dnn.readNetFromDarknet("D:/Coding/YOLO_Vision/Resources/yolov4-tiny.cfg", "D:/Coding/YOLO_Vision/Resources/yolov4-tiny.weights")
#     net = cv2.dnn.readNetFromDarknet("D:/Coding/YOLO_Vision/Resources/yolov4.cfg", "D:/Coding/YOLO_Vision/Resources/yolov4.weights")
#     layer_names= net.getLayerNames()
#     out_layers_indexes = net.getUnconnectedOutLayers()
#     # out_layers = [layer_names[index[0] - 1] for index in out_layers_indexes]
#     out_layers = [layer_names[index - 1] for index in out_layers_indexes]

#     with open("Resources/coco.names.txt") as file:
#         classes = file.read().split("\n")
    
#     classes_to_look_for = ["person"]

#     start_image_object_detection()
##############################




### For video handling
def start_video_object_detection():
    while True:
        video_camera_picture = cv2.VideoCapture(0)

        while video_camera_picture.isOpened():
            ret, frame = video_camera_picture.read()
            if not ret:
                break
        
            frame = apply_yolo_object_detection(frame)
            frame = cv2.resize(frame, (1920 // 2, 1080 // 2))
            cv2.imshow('Video Capture', frame)
            if cv2.waitKey(1) == ord('q'):
                win_close = True
                break
        if win_close:
                break

        video_camera_picture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    net = cv2.dnn.readNetFromDarknet("Resources/yolov4-tiny.cfg", "Resources/yolov4-tiny.weights")
    layer_names= net.getLayerNames()
    out_layers_indexes = net.getUnconnectedOutLayers()
    out_layers = [layer_names[index - 1] for index in out_layers_indexes]

    with open("Resources/coco.names.txt") as file:
        classes = file.read().split("\n")
    
    classes_to_look_for = ["person"]

    start_video_object_detection()

        