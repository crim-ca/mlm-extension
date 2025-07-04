from pystac import Item, Link
from stac_model.input import InputStructure, ModelInput
from stac_model.output import MLMClassification, ModelOutput, ModelResult
from stac_model.schema import ItemMLModelExtension, MLModelExtension, MLModelProperties

import torch.nn as nn
from typing import Optional


def get_input_channels(model: nn.Module) -> int:
    """
    Get input channels from the first Conv2d layer in the model.
    """
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            return layer.in_channels
    return 3  # default fallback


def get_output_channels(model: nn.Module) -> int:
    """
    Get output channels from the last Linear or Conv2d layer in the model.
    """
    for layer in reversed(list(model.modules())):
        if isinstance(layer, nn.Linear):
            return layer.out_features
        elif isinstance(layer, nn.Conv2d):
            return layer.out_channels
    return 10  # default fallback


def from_torch(
    model: nn.Module,
    *,
    weights: Optional[object] = None,
    item_id: str = "torch-model",
    bbox: Optional[list[float]] = None,
    geometry: Optional[dict] = None,
    links: Optional[list[dict]] = None,
    datetime_range: tuple[str, str] = (
        "1900-01-01T00:00:00Z",
        "9999-01-01T00:00:00Z",
    ),  # training data timestamp range par default voir papier
) -> ItemMLModelExtension:
    total_params = sum(p.numel() for p in model.parameters())
    arch = f"{model.__class__.__module__}.{model.__class__.__name__}"
    task = {"classification"}

    print("model passed:", vars(model))
    print(f"Modèle: {arch}, total params: {total_params}")

    if weights is not None and hasattr(weights, "meta"):
        in_chans = weights.meta.get("in_chans")
        num_classes = weights.meta.get("num_classes")
    else:
        in_chans = get_input_channels(model)
        num_classes = get_output_channels(model)

    input_shape = [1, in_chans, 224, 224]
    output_shape = [1, num_classes]

    print(f"Input shape: {input_shape}")
    print(f"Output shape: {output_shape}")

    input_struct = InputStructure(
        shape=input_shape,
        dim_order=["batch", "channel", "height", "width"],
        data_type="float32",
    )

    if weights is not None and hasattr(weights, "meta") and "bands" in weights.meta:
        bands = weights.meta["bands"]
    else:
        bands = [f"band_{i}" for i in range(input_shape[1])]

    print(f"Bands: {bands}")

    model_input = ModelInput(
        name="model_input",
        bands=bands,
        input=input_struct,
        resize_type=None,
        value_scaling=None,
        pre_processing_function=None,
    )

    classes = getattr(
        model,
        "classes",
        [
            MLMClassification(value=i, name=f"class_{i}", description=f"Auto-generated class {i}")
            for i in range(output_shape[-1])
        ],
    )
    print("classes", classes)

    model_output = ModelOutput(
        name="model_output",
        tasks=task,
        result=ModelResult(
            shape=output_shape,
            dim_order=["batch", "classes"],
            data_type="float32",
        ),
        classes=classes,
        post_processing_function=None,
    )

    mlm_props = MLModelProperties(
        name=item_id,
        architecture=arch,
        tasks=task,
        input=[model_input],
        output=[model_output],
        total_parameters=total_params,
        pretrained=True,
        pretrained_source=None,
    )

    bbox = bbox or [-7.88, 37.13, 27.91, 58.21]
    geometry = geometry or {
        "type": "Polygon",
        "coordinates": [
            [
                [-7.88, 37.13],
                [-7.88, 58.21],
                [27.91, 58.21],
                [27.91, 37.13],
                [-7.88, 37.13],
            ]
        ],
    }

    item = Item(
        id=item_id,
        geometry=geometry,
        bbox=bbox,
        datetime=None,
        properties={
            "start_datetime": datetime_range[0],
            "end_datetime": datetime_range[1],
            "description": "An Item with Machine Learning Model Extension metadata for a PyTorch model.",
        },
        stac_extensions=[MLModelExtension.get_schema_uri()],
    )

    for link in links or []:
        item.add_link(Link(**link))

    ext = MLModelExtension.ext(item, add_if_missing=True)
    ext.apply(mlm_props)

    return ItemMLModelExtension(item)
