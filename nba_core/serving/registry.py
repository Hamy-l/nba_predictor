"""Model registry for the module-1 scoring surface."""

from __future__ import annotations

from dataclasses import dataclass, field

from constants import MODEL_CATALOG, MODEL_TYPE_LABELS

__all__ = ["PipelineDescriptor", "ModelRegistry", "registry"]


@dataclass(frozen=True)
class PipelineDescriptor:
    """Immutable description of one selectable scoring pipeline."""

    key: str
    name: str
    kind: str
    description: str
    requires_sequence: bool
    estimator_family: str
    calibration: dict = field(default_factory=dict)

    @property
    def type_label(self) -> str:
        return MODEL_TYPE_LABELS.get(self.kind, self.kind)

    def as_public_dict(self) -> dict:
        """Serialisable view served to the browser."""
        return {
            "name": self.name,
            "type": self.kind,
            "description": self.description,
            "requires_sequence": self.requires_sequence,
        }


class ModelRegistry:
    """Keyed access to the selectable pipelines."""

    def __init__(self, catalog: dict | None = None):
        source = catalog if catalog is not None else MODEL_CATALOG
        self._pipelines: dict[str, PipelineDescriptor] = {}
        for key, spec in source.items():
            self._pipelines[key] = PipelineDescriptor(
                key=key,
                name=spec["name"],
                kind=spec["type"],
                description=spec["description"],
                requires_sequence=bool(spec.get("requires_sequence", False)),
                estimator_family=spec.get("estimator_family", "base"),
                calibration=dict(spec.get("calibration", {})),
            )

    def keys(self) -> list[str]:
        return list(self._pipelines.keys())

    def __contains__(self, key: object) -> bool:
        return key in self._pipelines

    def __iter__(self):
        return iter(self._pipelines)

    def items(self):
        return self._pipelines.items()

    def get(self, key: str) -> PipelineDescriptor:
        if key not in self._pipelines:
            raise KeyError(key)
        return self._pipelines[key]

    def public_catalog(self) -> dict:
        """Browser-facing catalogue, preserving catalogue order."""
        return {
            key: descriptor.as_public_dict()
            for key, descriptor in self._pipelines.items()
        }

    def type_label(self, kind: str) -> str:
        return MODEL_TYPE_LABELS.get(kind, kind)


registry = ModelRegistry()
