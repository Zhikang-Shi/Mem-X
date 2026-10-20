"""Lazy DINO feature and SAM-mask providers for the semantic builder."""

from __future__ import annotations

import contextlib
import importlib
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .geometry import masked_detection_position
from .models import Detection
from .vocabulary import RuntimeVocabulary


def _rgb_array(image: Any) -> np.ndarray:
    value = np.asarray(image)
    if value.ndim != 3 or value.shape[2] < 3:
        raise ValueError("RGB image must have shape HxWx3+")
    return np.asarray(value[..., :3], dtype=np.uint8)


class Dinov3FeatureExtractor:
    """Normalize model output while keeping torch/model loading lazy."""

    def __init__(
        self, model: Any, *, device: str = "cpu", image_size: int = 518,
        preprocess: Callable[[Any], Any] | None = None,
    ) -> None:
        if int(image_size) < 16:
            raise ValueError("image_size must be at least 16")
        self.model, self.device = model, str(device)
        self.image_size, self.preprocess = int(image_size), preprocess
        if hasattr(self.model, "eval"):
            self.model.eval()
        if hasattr(self.model, "to"):
            self.model.to(self.device)

    @classmethod
    def from_local_torch_hub(
        cls, repository: str | Path, model_name: str, *, device: str = "cpu",
        checkpoint: str | Path | None = None, model_kwargs: Mapping[str, Any] | None = None,
    ) -> "Dinov3FeatureExtractor":
        import torch

        model = torch.hub.load(
            str(repository), str(model_name), source="local", **dict(model_kwargs or {})
        )
        if checkpoint is not None:
            raw = torch.load(str(checkpoint), map_location="cpu")
            state = raw.get("state_dict", raw.get("model", raw)) if isinstance(raw, dict) else raw
            if not isinstance(state, dict):
                raise ValueError("DINO checkpoint does not contain a state dictionary")
            cleaned = {
                (str(key)[7:] if str(key).startswith("module.") else str(key)): value
                for key, value in state.items()
            }
            model.load_state_dict(cleaned, strict=False)
        return cls(model, device=device)

    @classmethod
    def from_huggingface_local(
        cls, model_path: str | Path, *, device: str = "cpu", image_size: int = 518,
        trust_remote_code: bool = True,
    ) -> "Dinov3FeatureExtractor":
        """Load a local Transformers DINO checkpoint without network access."""

        from transformers import AutoImageProcessor, AutoModel

        processor = AutoImageProcessor.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=trust_remote_code,
        )
        model = AutoModel.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=trust_remote_code,
        )

        def preprocess(image: Any) -> Mapping[str, Any]:
            return processor(
                images=image, return_tensors="pt", do_resize=True,
                size={"shortest_edge": int(image_size)}, do_center_crop=True,
                crop_size={"height": int(image_size), "width": int(image_size)},
                do_normalize=True, image_mean=[0.485, 0.456, 0.406],
                image_std=[0.229, 0.224, 0.225],
            )

        return cls(model, device=device, image_size=image_size, preprocess=preprocess)

    def _tensor(self, image: Any):
        import torch
        import torch.nn.functional as functional

        if self.preprocess is not None:
            tensor = self.preprocess(image)
            if isinstance(tensor, Mapping):
                return {
                    str(key): value if hasattr(value, "to") else torch.as_tensor(value)
                    for key, value in tensor.items()
                }
            if not hasattr(tensor, "to"):
                tensor = torch.as_tensor(tensor)
        else:
            array = _rgb_array(image).astype(np.float32) / 255.0
            tensor = torch.from_numpy(array).permute(2, 0, 1)
            tensor = functional.interpolate(
                tensor.unsqueeze(0), size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            ).squeeze(0)
            mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
            std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
            tensor = (tensor - mean) / std
        return tensor.unsqueeze(0) if tensor.ndim == 3 else tensor

    @staticmethod
    def _feature(output: Any) -> Any:
        if isinstance(output, Mapping):
            for key in ("x_norm_clstoken", "pooler_output", "features", "last_hidden_state"):
                if key in output and output[key] is not None:
                    output = output[key]
                    break
        if isinstance(output, (list, tuple)):
            output = output[0]
        if getattr(output, "ndim", 0) == 3:
            output = output[:, 0] if output.shape[1] > 1 else output.mean(dim=1)
        if getattr(output, "ndim", 0) > 2:
            output = output.flatten(start_dim=1)
        return output

    def __call__(self, image: Any) -> list[float]:
        import torch

        prepared = self._tensor(image)
        with torch.inference_mode():
            if isinstance(prepared, Mapping):
                output = self.model(**{
                    key: value.to(self.device) for key, value in prepared.items()
                })
            else:
                output = self.model(prepared.to(self.device))
            output = self._feature(output)
        feature = np.asarray(output.detach().float().cpu()).reshape(-1)
        norm = float(np.linalg.norm(feature))
        if not feature.size or not np.isfinite(feature).all() or norm <= 1.0e-12:
            raise ValueError("DINO model returned an invalid feature")
        return (feature / norm).tolist()


class Sam3LocalMaskPredictor:
    """Run a local SAM3 image model and return masks, boxes, and scores.

    SAM3's vendor package is imported lazily from ``source_root``.  This keeps
    the eight modules importable on CPU-only machines while providing a real
    deployment provider rather than depending on the legacy SG builder.
    """

    def __init__(
        self, source_root: str | Path, checkpoint: str | Path, *, device: str = "cpu",
        bpe_path: str | Path | None = None, threshold: float = 0.20,
        category_batch_size: int = 10,
    ) -> None:
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("SAM3 threshold must be in [0, 1]")
        if not 1 <= int(category_batch_size) <= 256:
            raise ValueError("SAM3 category_batch_size must be in [1, 256]")
        root = Path(source_root).resolve()
        checkpoint_path = Path(checkpoint).resolve()
        vocabulary_path = Path(bpe_path).resolve() if bpe_path else (
            root / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        )
        if not root.is_dir() or not checkpoint_path.is_file() or not vocabulary_path.is_file():
            raise FileNotFoundError("SAM3 source, checkpoint, and BPE vocabulary must exist")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        import torch

        sam3 = importlib.import_module("sam3")
        dataset = importlib.import_module("sam3.train.data.sam3_image_dataset")
        transforms = importlib.import_module("sam3.train.transforms.basic_for_api")
        collator = importlib.import_module("sam3.train.data.collator")
        misc = importlib.import_module("sam3.model.utils.misc")
        postprocessors = importlib.import_module("sam3.eval.postprocessors")
        self._torch, self._dataset = torch, dataset
        self._collate = collator.collate_fn_api
        self._copy_to_device = misc.copy_data_to_device
        self._postprocessor_type = postprocessors.PostProcessImage
        self.device = torch.device(str(device))
        self.threshold, self.category_batch_size = float(threshold), int(category_batch_size)
        self.model = sam3.build_sam3_image_model(
            bpe_path=str(vocabulary_path), checkpoint_path=str(checkpoint_path)
        )
        self.model.to(self.device)
        self._move_tensor_caches()
        self.model.eval()
        self.transform = transforms.ComposeAPI([
            transforms.RandomResizeAPI(
                sizes=1008, max_size=1008, square=True, consistent_transform=False
            ),
            transforms.ToTensorAPI(),
            transforms.NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self._query_counter = 1

    def _move_tensor_caches(self) -> None:
        torch = self._torch

        def move(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                return value.to(self.device)
            if isinstance(value, dict):
                return {key: move(item) for key, item in value.items()}
            if isinstance(value, tuple):
                return tuple(move(item) for item in value)
            if isinstance(value, list):
                return [move(item) for item in value]
            return value

        for module in self.model.modules():
            for name in ("cache", "coord_cache", "compilable_cord_cache"):
                if hasattr(module, name):
                    setattr(module, name, move(getattr(module, name)))

    def _datapoint(self, image: Any, labels: Sequence[str]):
        from PIL import Image

        pil = image if isinstance(image, Image.Image) else Image.fromarray(_rgb_array(image))
        dataset = self._dataset
        width, height = pil.size
        datapoint = dataset.Datapoint(
            find_queries=[], images=[dataset.Image(data=pil, objects=[], size=[height, width])]
        )
        query_ids = {}
        for label in labels:
            query_id = self._query_counter
            self._query_counter += 1
            datapoint.find_queries.append(dataset.FindQueryLoaded(
                query_text=str(label), image_id=0, object_ids_output=[], is_exhaustive=True,
                query_processing_order=0,
                inference_metadata=dataset.InferenceMetadata(
                    coco_image_id=query_id, original_image_id=query_id,
                    original_category_id=1, original_size=[height, width],
                    object_id=0, frame_index=0,
                ),
            ))
            query_ids[str(label)] = query_id
        return self.transform(datapoint), query_ids

    def _batch(self, image: Any, labels: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
        torch = self._torch
        datapoint, query_ids = self._datapoint(image, labels)
        batch = self._collate([datapoint], dict_key="memory_module")["memory_module"]
        batch = self._copy_to_device(batch, self.device, non_blocking=True)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
            autocast = torch.autocast("cuda", dtype=torch.bfloat16)
        else:
            autocast = contextlib.nullcontext()
        with torch.inference_mode(), autocast:
            output = self.model(batch)
        processor = self._postprocessor_type(
            max_dets_per_img=-1, iou_type="segm", use_original_sizes_box=True,
            use_original_sizes_mask=True, convert_mask_to_rle=False,
            detection_threshold=self.threshold, to_cpu=True,
        )
        processed = processor.process_results(output, batch.find_metadatas)
        result: dict[str, list[dict[str, Any]]] = {}
        for label, query_id in query_ids.items():
            value = processed.get(query_id)
            if value is None or value.get("boxes") is None:
                result[label] = []
                continue
            boxes = np.asarray(value["boxes"].float())
            scores = np.asarray(value["scores"].float())
            masks = np.asarray(value["masks"].float())
            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks[:, 0]
            result[label] = [
                {"mask": masks[index] > 0, "score": float(scores[index]),
                 "bbox_xyxy": boxes[index].tolist()}
                for index in range(min(len(boxes), len(scores), len(masks)))
            ]
        return result

    def __call__(self, image: Any, labels: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
        result = {str(label): [] for label in labels}
        for start in range(0, len(labels), self.category_batch_size):
            result.update(self._batch(image, labels[start:start + self.category_batch_size]))
        return result


class Sam3DetectionProvider:
    """Convert SAM-style masks into validated 3-D ``Detection`` records.

    ``predict_masks`` receives ``(rgb, labels)`` and may return either a flat
    sequence of records or ``{label: [record_or_mask, ...]}``.  A record may
    contain ``mask``, ``score`` and ``bbox_xyxy``.
    """

    def __init__(
        self, predict_masks: Callable[[Any, Sequence[str]], Any],
        feature_extractor: Callable[[Any], Sequence[float]],
        vocabulary: RuntimeVocabulary, intrinsics: Mapping[str, float],
        *, minimum_score: float = 0.0,
        rgb_path: Callable[[int], str] | None = None,
        crop_path: Callable[[int, str, int], str | None] | None = None,
    ) -> None:
        self.predict_masks, self.feature_extractor = predict_masks, feature_extractor
        self.vocabulary, self.intrinsics = vocabulary, dict(intrinsics)
        self.minimum_score = float(minimum_score)
        self.rgb_path, self.crop_path = rgb_path, crop_path
        if not 0.0 <= self.minimum_score <= 1.0:
            raise ValueError("minimum_score must be in [0, 1]")
        for key in ("fx", "fy", "cx", "cy"):
            if key not in self.intrinsics:
                raise ValueError(f"missing camera intrinsic {key}")

    @staticmethod
    def _bbox(mask: np.ndarray) -> list[int] | None:
        rows, cols = np.where(mask)
        if not rows.size:
            return None
        return [int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1]

    def _records(self, raw: Any) -> Iterable[tuple[str, Mapping[str, Any]]]:
        if isinstance(raw, Mapping):
            if "label" in raw and "mask" in raw:
                yield str(raw["label"]), raw
                return
            for label, values in raw.items():
                entries = values if isinstance(values, (list, tuple)) else [values]
                for entry in entries:
                    yield str(label), entry if isinstance(entry, Mapping) else {"mask": entry}
            return
        if isinstance(raw, (list, tuple)):
            for entry in raw:
                if not isinstance(entry, Mapping) or not entry.get("label"):
                    raise ValueError("flat SAM records require label and mask")
                yield str(entry["label"]), entry
            return
        raise ValueError("SAM predictor returned an unsupported result")

    def __call__(
        self, rgb: Any, depth: Any, camera_to_world: Any, frame_index: int
    ) -> list[Detection]:
        image, result = _rgb_array(rgb), []
        raw = self.predict_masks(image, self.vocabulary.labels)
        per_label_index: dict[str, int] = {}
        for raw_label, record in self._records(raw):
            label = self.vocabulary.canonical(raw_label)
            if label not in self.vocabulary.labels:
                continue
            score = float(record.get("score", 1.0))
            if not np.isfinite(score) or score < self.minimum_score:
                continue
            mask = np.asarray(record.get("mask"), dtype=bool)
            if mask.shape != image.shape[:2]:
                raise ValueError(f"SAM mask for {label!r} has shape {mask.shape}")
            bbox = record.get("bbox_xyxy")
            if bbox is None:
                bbox = self._bbox(mask)
            if bbox is None or not self.vocabulary.accepts_box(label, bbox):
                continue
            x1, y1, x2, y2 = map(int, bbox)
            crop = image[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            if not crop.size:
                continue
            position = masked_detection_position(
                np.asarray(depth), bbox, np.asarray(camera_to_world), self.intrinsics, mask
            )
            index = per_label_index.get(label, 0)
            per_label_index[label] = index + 1
            result.append(Detection(
                label=label, score=min(1.0, max(0.0, score)),
                feature=self.feature_extractor(crop), position_xyz=position,
                bbox_xyxy=bbox,
                rgb_path=self.rgb_path(int(frame_index)) if self.rgb_path else "",
                crop_path=self.crop_path(int(frame_index), label, index) if self.crop_path else None,
                support={"mask_area_px": int(np.count_nonzero(mask))},
                detector_id=f"{frame_index}:{label}:{index}",
            ))
        return result
