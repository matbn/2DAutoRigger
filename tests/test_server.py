"""The HTTP service, which is the whole JavaScript surface."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rigger2d.server.app import create_app


@pytest.fixture()
def client(tmp_path: Path, character_image: Path) -> TestClient:
    app = create_app(projects=tmp_path / "projects")
    return TestClient(app)


def _upload(client: TestClient, image: Path) -> str:
    response = client.post(
        "/api/projects/upload",
        files={"file": (image.name, image.read_bytes(), "image/png")},
    )
    assert response.status_code == 200, response.text
    return response.json()["id"]


def test_health_reports_the_stages(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "parts" in body["stages"]
    assert body["refine_gate"] == "parts"


def test_the_backends_are_listed(client: TestClient) -> None:
    response = client.get("/api/backends")
    assert response.status_code == 200
    backends = response.json()["backends"]
    assert any(backend["id"] == "core.fallback" for backend in backends)
    for backend in backends:
        assert backend["status"]


def test_the_templates_are_listed(client: TestClient) -> None:
    response = client.get("/api/templates")
    assert response.status_code == 200
    templates = response.json()["templates"]
    assert any(template["id"] == "humanoid.biped.v1" for template in templates)


def test_a_character_can_be_uploaded_and_rigged(client: TestClient, character_image: Path) -> None:
    project_id = _upload(client, character_image)

    response = client.post(f"/api/projects/{project_id}/run", json={"until": "refine"})
    assert response.status_code == 200, response.text
    document = response.json()["document"]
    assert document["parts"] is not None
    assert document["rig"] is None, "the pipeline must stop at the refine gate"

    response = client.post(
        f"/api/projects/{project_id}/slice",
        json={"targets": ["native", "godot", "spine"], "skinning": "hybrid"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["document"]["rig"] is not None
    assert body["exports"]


def test_a_correction_reaches_the_server(client: TestClient, character_image: Path) -> None:
    project_id = _upload(client, character_image)
    client.post(f"/api/projects/{project_id}/run", json={"until": "refine"})

    response = client.post(
        f"/api/projects/{project_id}/refine",
        json={
            "edits": [
                {
                    "mask": {
                        "op": "add",
                        "part_id": "torso",
                        "points": [{"x": 10, "y": 10}],
                        "radius": 4,
                    },
                    "joint": None,
                }
            ]
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["edits"]["mask_add"] == 1

    response = client.get(f"/api/projects/{project_id}")
    assert response.json()["document"]["edits"]


def test_a_joint_can_be_moved(client: TestClient, character_image: Path) -> None:
    project_id = _upload(client, character_image)
    client.post(f"/api/projects/{project_id}/run", json={"until": "refine"})

    before = client.get(f"/api/projects/{project_id}").json()["document"]["landmarks"]
    response = client.post(
        f"/api/projects/{project_id}/joints",
        json={"edits": [{"joint": {"joint_id": "neck", "dx": 4, "dy": 2}, "mask": None}]},
    )
    assert response.status_code == 200, response.text
    after = response.json()["landmarks"]
    assert after["neck"]["x"] == pytest.approx(before["neck"]["x"] + 4)


def test_the_backends_can_be_compared(client: TestClient, character_image: Path) -> None:
    project_id = _upload(client, character_image)
    client.post(f"/api/projects/{project_id}/run", json={"until": "refine"})

    response = client.post(f"/api/projects/{project_id}/compare", json={"task": "parts"})
    assert response.status_code == 200, response.text
    attempts = response.json()["attempts"]
    assert attempts
    assert any(attempt["adapter_id"] == "core.fallback" for attempt in attempts)


def test_the_projects_are_listed(client: TestClient, character_image: Path) -> None:
    project_id = _upload(client, character_image)
    response = client.get("/api/projects")
    assert response.status_code == 200
    assert any(project["id"] == project_id for project in response.json()["projects"])


def test_an_unknown_project_is_a_404(client: TestClient) -> None:
    assert client.get("/api/projects/nope").status_code == 404


def test_a_project_can_be_deleted(client: TestClient, character_image: Path) -> None:
    project_id = _upload(client, character_image)
    assert client.delete(f"/api/projects/{project_id}").status_code == 200
    assert client.get(f"/api/projects/{project_id}").status_code == 404


def test_an_image_can_be_sent_as_base64(client: TestClient, character_image: Path) -> None:
    encoded = base64.b64encode(character_image.read_bytes()).decode("ascii")
    response = client.post("/api/projects", json={"image_base64": encoded})
    assert response.status_code == 200, response.text
    assert response.json()["document"]["image"]["width"] == 512


def test_the_browser_field_names_are_accepted(client: TestClient, character_image: Path) -> None:
    """The browser client sends camelCase; it must not 400 over the field name.

    Everything else the client sends is snake_case, but this one field was
    camelCase on the JavaScript side, so the field was silently dropped and the
    request failed with "give an image_path or image_base64".
    """

    encoded = base64.b64encode(character_image.read_bytes()).decode("ascii")
    response = client.post("/api/projects", json={"imageBase64": encoded})
    assert response.status_code == 200, response.text
    assert response.json()["document"]["image"]["width"] == 512


def test_a_project_without_an_image_is_rejected(client: TestClient) -> None:
    assert client.post("/api/projects", json={}).status_code == 400


def test_an_export_can_be_downloaded(client: TestClient, character_image: Path) -> None:
    project_id = _upload(client, character_image)
    client.post(f"/api/projects/{project_id}/run", json={"until": "refine"})
    client.post(
        f"/api/projects/{project_id}/slice",
        json={"targets": ["native"], "skinning": "hybrid"},
    )

    response = client.get(f"/api/projects/{project_id}/exports/rig.json")
    assert response.status_code == 200
    assert response.json()["bones"]


def test_a_download_cannot_escape_the_export_folder(client: TestClient, character_image: Path) -> None:
    project_id = _upload(client, character_image)
    response = client.get(f"/api/projects/{project_id}/exports/..%2F..%2Fdocument.json")
    assert response.status_code in (400, 404)
