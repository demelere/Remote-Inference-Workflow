import cv2
import json
import numpy as np
from typing import Dict, List, Tuple

class PredictionHandler:
    def __init__(self):
        """Initialize prediction handler with default visualization settings."""
        self.colors = {}  # Cache for class colors
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.thickness = 2

    def parse_prediction(self, data: bytes) -> Dict:
        """Parse prediction data from bytes to dictionary."""
        try:
            return json.loads(data.decode('utf-8'))
        except json.JSONDecodeError as e:
            print(f"Error parsing prediction: {e}")
            return {}
        except Exception as e:
            print(f"Unexpected error parsing prediction: {e}")
            return {}

    def _get_color(self, class_name: str) -> Tuple[int, int, int]:
        """Get consistent color for a class."""
        if class_name not in self.colors:
            # Generate random color for new class
            self.colors[class_name] = tuple(np.random.randint(0, 255, 3).tolist())
        return self.colors[class_name]

    def draw_predictions(self, frame: np.ndarray, predictions: Dict) -> np.ndarray:
        """Draw predictions on frame."""
        if not predictions or 'predictions' not in predictions:
            return frame

        frame_height, frame_width = frame.shape[:2]
        annotated_frame = frame.copy()

        for pred in predictions['predictions']:
            # Extract prediction data
            x = pred.get('x', 0)
            y = pred.get('y', 0)
            width = pred.get('width', 0)
            height = pred.get('height', 0)
            class_name = pred.get('class', 'unknown')
            confidence = pred.get('confidence', 0)

            # Convert normalized coordinates to pixel coordinates
            x1 = int((x - width/2) * frame_width)
            y1 = int((y - height/2) * frame_height)
            x2 = int((x + width/2) * frame_width)
            y2 = int((y + height/2) * frame_height)

            # Get color for class
            color = self._get_color(class_name)

            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, self.thickness)

            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, self.font, self.font_scale, self.thickness)
            y1_label = max(y1, label_size[1])
            cv2.rectangle(annotated_frame, (x1, y1_label - label_size[1]),
                        (x1 + label_size[0], y1_label + 5), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1_label),
                       self.font, self.font_scale, (255, 255, 255), 1)

        return annotated_frame

    def get_prediction_summary(self, predictions: Dict) -> str:
        """Get a summary of predictions for logging."""
        if not predictions or 'predictions' not in predictions:
            return "No predictions"

        num_predictions = len(predictions['predictions'])
        classes = {}
        for pred in predictions['predictions']:
            class_name = pred.get('class', 'unknown')
            classes[class_name] = classes.get(class_name, 0) + 1

        summary = f"Found {num_predictions} objects: "
        summary += ", ".join(f"{count} {cls}" for cls, count in classes.items())
        return summary 