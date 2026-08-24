from __future__ import annotations

from pathlib import Path

from diffusers_3d import BackendLicenseClass, IntegrationManifest3D

FAMILY_ROOT = Path(__file__).parents[3] / "src" / "diffusers_3d" / "families" / "hunyuan3d"


def test_manifest_revision_license_and_unmeasured_pipeline_parity():
    manifest = IntegrationManifest3D.load(FAMILY_ROOT / "diffusers_3d_integration.json")
    assert manifest.upstream.revision == "82920d643c0dc2f7bfd7255f45f62d386edfe60c"
    assert manifest.licenses.model.classification is BackendLicenseClass.RESTRICTED
    licenses = {artifact.artifact: artifact.license for artifact in manifest.licenses.artifacts}
    assert licenses["hunyuan-derived-code"].classification is BackendLicenseClass.RESTRICTED
    assert licenses["package-glue-code"].classification is BackendLicenseClass.PERMISSIVE
    pipeline = next(component for component in manifest.components if component.role == "pipeline")
    assert pipeline.parity[0].passed is False
    assert (FAMILY_ROOT / "LICENSE-TENCENT-HUNYUAN-3D-2.1").is_file()
    assert (FAMILY_ROOT / "NOTICE").is_file()
