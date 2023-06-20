import cv2

from ultralytics import YOLO

import supervision as sv

from roboflow import Roboflow


def main():

    model = YOLO("yolov8n.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5
    )

    for result in model.track(source="Bird1.mp4", stream=True, classes=14):
        frame = result.orig_img
        detections = sv.Detections.from_yolov8(result)

        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for  _, _, confidence, class_id, _
            in detections
        ]

        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

        cv2.imshow("Bird tracking", frame)

        if cv2.waitKey(30) == 27:
            break


if __name__ == "__main__":
    main()