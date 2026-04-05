"""
Drone Detection Pipeline — YOLOv8 + SAHI Slicing
==================================================
Part of the R-MMAF detection stack.

Uses SAHI's sliced inference to prevent small drone pixels from being
discarded during standard convolution downsampling. Wraps YOLOv8 with
a sliding window approach for high-resolution aerial imagery.

Author: Sanchit Agarwal
"""

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


def build_detection_model(
    model_path: str = "yolov8n.pt",
    confidence_threshold: float = 0.25,
    device: str = "cuda:0"
) -> AutoDetectionModel:
    """
    Loads YOLOv8 inside SAHI's sliced inference engine.

    Parameters
    ----------
    model_path : str
        Path to YOLOv8 weights (.pt file)
    confidence_threshold : float
        Minimum detection confidence score
    device : str
        Inference device — "cuda:0" or "cpu"

    Returns
    -------
    AutoDetectionModel
        SAHI-wrapped YOLOv8 model ready for sliced inference
    """
    return AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        device=device,
    )


def run_sliced_inference(
    image_path: str,
    detection_model: AutoDetectionModel,
    slice_height: int = 320,
    slice_width: int = 320,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
):
    """
    Runs sliced inference on an aerial image.

    The image is divided into overlapping tiles (slices). YOLOv8 runs
    independently on each tile, then detections are merged with NMS.
    This prevents tiny drone objects from disappearing at full resolution.

    Parameters
    ----------
    image_path : str
        Path to input image
    detection_model : AutoDetectionModel
        SAHI-wrapped model from build_detection_model()
    slice_height / slice_width : int
        Tile dimensions in pixels
    overlap_height_ratio / overlap_width_ratio : float
        Fractional overlap between adjacent tiles

    Returns
    -------
    PredictionResult
        SAHI result object (call .export_visuals() to save output)
    """
    return get_sliced_prediction(
        image_path,
        detection_model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
    )
