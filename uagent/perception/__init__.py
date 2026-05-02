"""Perception layer: MC Dropout + deterministic YOLO + calibration."""

from uagent.perception.posterior import Detection, GateDecision, Posterior

__all__ = ["Detection", "GateDecision", "Posterior"]
