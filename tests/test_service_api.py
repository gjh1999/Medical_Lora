from pathlib import Path

from fastapi.testclient import TestClient

from service_app import (
    IntentResponse,
    InvalidModelOutputError,
    ServiceSettings,
    create_app,
    normalize_model_payload,
)


VALID_RESULT = {
    "routing": {
        "need_record": True,
        "need_advice": True,
        "need_plan": False,
        "need_reminder": False,
        "need_history": False,
    },
    "confidence": {
        "need_record": 0.98,
        "need_advice": 0.95,
        "need_plan": 0.02,
        "need_reminder": 0.01,
        "need_history": 0.01,
    },
    "slots": {
        "record": {
            "blood_pressure": {
                "systolic": 145,
                "diastolic": 92,
                "unit": "mmHg",
                "time": "今天早上",
            }
        }
    },
    "missing_fields": [],
    "risk_flags": ["high_blood_pressure"],
    "normalized": {"date": None, "time_of_day": "morning", "units": {}},
}


class FakeRuntime:
    ready = True

    def load(self):
        return None

    def predict(self, text: str) -> IntentResponse:
        assert text
        return IntentResponse.model_validate(VALID_RESULT)


class InvalidOutputRuntime(FakeRuntime):
    def predict(self, text: str) -> IntentResponse:
        raise InvalidModelOutputError("模型未返回符合 Schema 的 JSON")


class LoadingRuntime(FakeRuntime):
    ready = False


def make_settings(api_key: str = "") -> ServiceSettings:
    return ServiceSettings(
        base_model_path="unused-in-api-tests",
        lora_path=Path("unused-in-api-tests"),
        api_key=api_key,
    )


def make_client(runtime=None, api_key: str = "") -> TestClient:
    app = create_app(
        runtime=runtime or FakeRuntime(),
        settings=make_settings(api_key),
        load_model_on_startup=False,
    )
    return TestClient(app)


def test_healthz_reports_ready_model():
    with make_client() as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model": "qwen2.5-7b-routing-lora"}


def test_healthz_returns_503_while_model_is_not_ready():
    with make_client(LoadingRuntime()) as client:
        response = client.get("/healthz")

    assert response.status_code == 503
    assert response.json() == {"status": "loading"}


def test_parse_returns_contract_fields():
    with make_client() as client:
        response = client.post(
            "/v1/intent/parse",
            json={"text": "记录血压145/92，这正常吗？", "user_id": "test-user"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {
        "routing",
        "confidence",
        "slots",
        "missing_fields",
        "risk_flags",
        "normalized",
    }
    assert payload["routing"]["need_record"] is True
    assert payload["routing"]["need_advice"] is True


def test_parse_rejects_blank_text():
    with make_client() as client:
        response = client.post("/v1/intent/parse", json={"text": "   "})

    assert response.status_code == 422


def test_parse_requires_configured_bearer_token():
    with make_client(api_key="secret") as client:
        missing = client.post("/v1/intent/parse", json={"text": "记录血压145/92"})
        invalid = client.post(
            "/v1/intent/parse",
            headers={"Authorization": "Bearer wrong"},
            json={"text": "记录血压145/92"},
        )
        valid = client.post(
            "/v1/intent/parse",
            headers={"Authorization": "Bearer secret"},
            json={"text": "记录血压145/92"},
        )

    assert missing.status_code == 401
    assert invalid.status_code == 401
    assert valid.status_code == 200


def test_parse_maps_invalid_model_output_to_502():
    with make_client(InvalidOutputRuntime()) as client:
        response = client.post("/v1/intent/parse", json={"text": "测试"})

    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "INVALID_MODEL_OUTPUT"


def test_normalize_historical_slot_variants():
    payload = {
        "slots": {
            "reminder": {"task": "测血糖", "time": "明早8点", "recurrence": None},
            "plan": {"preferences": None, "constraints": None},
            "advice": {"related_metrics": None},
        },
        "missing_fields": ["reminder.time", "reminder.unknown"],
    }

    normalized = normalize_model_payload(payload)

    assert normalized["slots"]["reminder"]["repeat"] is None
    assert "recurrence" not in normalized["slots"]["reminder"]
    assert normalized["slots"]["plan"]["preferences"] == []
    assert normalized["slots"]["plan"]["constraints"] == []
    assert normalized["slots"]["advice"]["related_metrics"] == []
    assert normalized["missing_fields"] == ["reminder.unknown"]
