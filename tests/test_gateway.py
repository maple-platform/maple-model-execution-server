import unittest
import os
import tempfile
import json
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

from app.model_config import load_model_config, validate_all_model_configs
from main import InferRequest, InferV2Request, infer, infer_v2, ready


class _Response:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "status": "ok",
            "result": {
                "result_type": "classification",
                "predictions": [],
                "images_b64": [],
            },
        }


class _AsyncClient:
    last_timeout = None
    last_url = None
    last_json = None

    def __init__(self, *, timeout):
        type(self).last_timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def post(self, _url, *, json):
        type(self).last_url = _url
        type(self).last_json = json
        return _Response()


class _ReadyClient(_AsyncClient):
    async def get(self, _url):
        return _Response()


class LegacyInferTest(unittest.IsolatedAsyncioTestCase):
    async def test_default_timeout_does_not_require_v2_model_config(self):
        request = InferRequest(
            model_id="Pulmonology/test",
            input_data="/data/test.dcm",
            params={
                "container_url": "http://runtime-yolo:8000",
                "container_endpoint": "/run/v2",
                "model_name": "test",
            },
        )
        with patch("main.httpx.AsyncClient", _AsyncClient):
            response = await infer(request)
        self.assertEqual(_AsyncClient.last_timeout, 120.0)
        self.assertEqual(response.metadata["result_type"], "classification")

    async def test_explicit_timeout_is_respected(self):
        request = InferRequest(
            model_id="Pulmonology/test",
            input_data="/data/test.dcm",
            params={
                "container_url": "http://runtime-yolo:8000",
                "model_name": "test",
                "timeout": 42,
            },
        )
        with patch("main.httpx.AsyncClient", _AsyncClient):
            await infer(request)
        self.assertEqual(_AsyncClient.last_timeout, 42.0)

    async def test_placeholder_url_is_rejected_without_http_call(self):
        request = InferRequest(
            model_id="Orthopedics/FracAtlas_Fracture_Fusion",
            input_data="/data/image.png",
            params={"container_url": "관리자 작성 예정"},
        )
        with patch("main.httpx.AsyncClient") as client:
            with self.assertRaises(HTTPException) as caught:
                await infer(request)
        self.assertEqual(caught.exception.status_code, 422)
        self.assertEqual(caught.exception.detail["error_type"], "invalid_container_url")
        client.assert_not_called()


class InferV2Test(unittest.IsolatedAsyncioTestCase):
    models_dir = str(Path(__file__).resolve().parents[1] / "models")

    def setUp(self):
        self.env = patch.dict(
            os.environ,
            {
                "MODELS_DIR": self.models_dir,
                "RUNTIME_BASIC_URL": "http://runtime-basic:8000",
                "RUNTIME_MEDICAL_URL": "http://runtime-medical:8000",
                "RUNTIME_YOLO_URL": "http://runtime-yolo:8000",
                "RUNTIME_NNUNET_URL": "http://runtime-nnunet:8000",
            },
        )
        self.env.start()

    def tearDown(self):
        self.env.stop()

    def test_required_model_runtime_mapping(self):
        expected = {
            "FracAtlas_Fracture_Fusion": "runtime-medical",
            "AASCE_Scoliosis_Cobb": "runtime-medical",
            "RSNA_Pneumonia_YOLO26x": "runtime-yolo",
            "VerSe_Vertebrae_CT": "runtime-nnunet",
        }
        for model_name, runtime in expected.items():
            with self.subTest(model_name=model_name):
                self.assertEqual(load_model_config(model_name)["runtime"], runtime)

    async def test_selects_runtime_without_caller_container_url(self):
        request = InferV2Request(
            model_name="FracAtlas_Fracture_Fusion",
            input_path="/data/image.png",
            output_dir="/data/output/s1",
            params={"container_url": "http://attacker.invalid"},
        )
        with patch("main.httpx.AsyncClient", _AsyncClient):
            response = await infer_v2(request)
        self.assertEqual(_AsyncClient.last_url, "http://runtime-medical:8000/run/v2")
        self.assertEqual(response.metadata["runtime"], "runtime-medical")
        self.assertEqual(response.result["predictions"], [])
        self.assertNotIn("container_url", _AsyncClient.last_json)

    async def test_unknown_model_returns_404(self):
        with self.assertRaises(HTTPException) as caught:
            await infer_v2(InferV2Request(model_name="does-not-exist", input_path="/data/a"))
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.detail["error_type"], "model_not_found")

    async def test_unknown_runtime_returns_422(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory) / "bad-runtime"
            model_dir.mkdir()
            (model_dir / "config.yaml").write_text(
                "model_name: bad-runtime\n"
                "execution_mode: external_runtime\n"
                "runtime: runtime-unknown\n"
                "runner_path: /app/models/bad-runtime/runner.py\n"
                "inference_path: /app/models/bad-runtime/inference.py\n"
                "model_path: /app/models/bad-runtime/model.pt\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"MODELS_DIR": directory}):
                with self.assertRaises(HTTPException) as caught:
                    await infer_v2(InferV2Request(model_name="bad-runtime", input_path="/data/a"))
        self.assertEqual(caught.exception.status_code, 422)
        self.assertEqual(caught.exception.detail["error_type"], "unsupported_runtime")

    async def test_missing_or_invalid_runtime_url_returns_503(self):
        for value in (None, "관리자 작성 예정", "runtime-medical:8000", "   "):
            env = os.environ.copy()
            if value is None:
                env.pop("RUNTIME_MEDICAL_URL", None)
            else:
                env["RUNTIME_MEDICAL_URL"] = value
            with self.subTest(value=value), patch.dict(os.environ, env, clear=True):
                with self.assertRaises(HTTPException) as caught:
                    await infer_v2(
                        InferV2Request(model_name="FracAtlas_Fracture_Fusion", input_path="/data/a")
                    )
            self.assertEqual(caught.exception.status_code, 503)

    def test_all_model_configs_are_valid(self):
        report = validate_all_model_configs()
        self.assertEqual(
            {key: report[key] for key in ("total", "valid", "invalid")},
            {"total": 73, "valid": 73, "invalid": 0},
            report["errors"],
        )

    def test_meta_service_urls_match_model_runtimes(self):
        runtime_urls = {
            "runtime-basic": "http://runtime-basic:8000",
            "runtime-medical": "http://runtime-medical:8000",
            "runtime-yolo": "http://runtime-yolo:8000",
            "runtime-nnunet": "http://runtime-nnunet:8000",
        }
        root = Path(__file__).resolve().parents[1]
        meta_paths = {
            path.parent.name: path
            for path in root.glob("AI_Models/*/*/meta.json")
        }
        for model_name in (
            path.parent.name
            for path in (root / "models").glob("*/config.yaml")
            if path.parent.name != "example_model"
        ):
            with self.subTest(model_name=model_name):
                config = load_model_config(model_name)
                metadata = json.loads(meta_paths[model_name].read_text(encoding="utf-8"))
                self.assertEqual(
                    metadata["docker"]["service_url"],
                    runtime_urls[config["runtime"]],
                )

    async def test_readiness_checks_all_runtime_health_endpoints(self):
        with patch("main.httpx.AsyncClient", _ReadyClient):
            response = await ready()
        self.assertEqual(response["status"], "ready")
        self.assertEqual(response["models"], {"total": 73, "valid": 73, "invalid": 0})
        self.assertEqual(set(response["runtimes"].values()), {"ready"})


if __name__ == "__main__":
    unittest.main()
