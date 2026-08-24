from __future__ import annotations

from dataclasses import replace

import pytest

from diffusers_3d import (
    Object3DModelRegistration,
    Object3DModelRegistry,
    Object3DPipelineRegistration,
    Object3DPipelineRegistry,
    Object3DRegistrationError,
)


def test_model_registry_resolves_only_exact_reviewed_metadata(
    tiny_model_class,
    other_tiny_model_class,
    tiny_model_subclass,
):
    metadata = tiny_model_class.object3d_metadata()
    registration = Object3DModelRegistration(tiny_model_class, metadata)
    registry = Object3DModelRegistry((registration,))

    assert registry.resolve(metadata) is tiny_model_class
    with pytest.raises(Object3DRegistrationError, match="already registered"):
        registry.register(registration)
    with pytest.raises(Object3DRegistrationError, match="does not match"):
        registry.resolve(replace(metadata, family_id="different-family"))
    with pytest.raises(Object3DRegistrationError, match="no exact reviewed"):
        registry.resolve(other_tiny_model_class.object3d_metadata())
    with pytest.raises(Object3DRegistrationError, match="no exact reviewed"):
        registry.resolve(tiny_model_subclass.object3d_metadata())


def test_model_registry_rejects_ambiguous_and_unreviewed_registration(
    tiny_model_class,
    other_tiny_model_class,
    unreviewed_model_class,
):
    registry = Object3DModelRegistry(
        (Object3DModelRegistration(tiny_model_class, tiny_model_class.object3d_metadata()),)
    )

    with pytest.raises(Object3DRegistrationError, match="Ambiguous"):
        registry.register(
            Object3DModelRegistration(
                other_tiny_model_class,
                other_tiny_model_class.object3d_metadata(),
            )
        )
    with pytest.raises(Object3DRegistrationError, match="explicitly reviewed"):
        Object3DModelRegistry().register(
            Object3DModelRegistration(
                unreviewed_model_class,
                unreviewed_model_class.object3d_metadata(),
            )
        )


def test_pipeline_registry_resolves_exact_class_family_task_and_metadata(
    tiny_pipeline_class,
    other_tiny_pipeline_class,
    tiny_pipeline_subclass,
):
    metadata = tiny_pipeline_class.object3d_model_index()
    registration = Object3DPipelineRegistration(tiny_pipeline_class, metadata)
    registry = Object3DPipelineRegistry((registration,))

    assert registry.resolve(metadata, "text-to-3d") is tiny_pipeline_class
    with pytest.raises(Object3DRegistrationError, match="already registered"):
        registry.register(registration)
    with pytest.raises(Object3DRegistrationError, match="does not match"):
        registry.resolve(replace(metadata, required_backends=()), "text-to-3d")
    with pytest.raises(Object3DRegistrationError, match="no exact reviewed"):
        registry.resolve(other_tiny_pipeline_class.object3d_model_index(), "text-to-3d")
    with pytest.raises(Object3DRegistrationError, match="no exact reviewed"):
        registry.resolve(tiny_pipeline_subclass.object3d_model_index(), "text-to-3d")
    with pytest.raises(Object3DRegistrationError, match="no exact reviewed"):
        registry.resolve(
            replace(metadata, pipeline_class="unregistered.module.TinyObject3DPipeline"),
            "text-to-3d",
        )


def test_pipeline_registry_rejects_ambiguous_and_mismatched_registration(
    tiny_pipeline_class,
    other_tiny_pipeline_class,
):
    registry = Object3DPipelineRegistry(
        (
            Object3DPipelineRegistration(
                tiny_pipeline_class,
                tiny_pipeline_class.object3d_model_index(),
            ),
        )
    )

    with pytest.raises(Object3DRegistrationError, match="Ambiguous"):
        registry.register(
            Object3DPipelineRegistration(
                other_tiny_pipeline_class,
                other_tiny_pipeline_class.object3d_model_index(),
            )
        )
    with pytest.raises(Object3DRegistrationError, match="differing fields"):
        Object3DPipelineRegistry().register(
            Object3DPipelineRegistration(
                tiny_pipeline_class,
                replace(tiny_pipeline_class.object3d_model_index(), family_id="other-family"),
            )
        )


def test_registry_freeze_is_read_only(tiny_pipeline_class):
    registry = Object3DPipelineRegistry().freeze()
    registration = Object3DPipelineRegistration(
        tiny_pipeline_class,
        tiny_pipeline_class.object3d_model_index(),
    )

    with pytest.raises(Object3DRegistrationError, match="read-only"):
        registry.register(registration)
