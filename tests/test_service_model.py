import os

import pytest

from service_app import RoutingRuntime, ServiceSettings


pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def runtime():
    if os.getenv("RUN_MODEL_TESTS") != "1":
        pytest.skip("set RUN_MODEL_TESTS=1 to load the real routing model")
    model_runtime = RoutingRuntime(ServiceSettings.from_env())
    model_runtime.load()
    return model_runtime


@pytest.mark.parametrize(
    ("text", "expected_flags"),
    [
        ("明早8点提醒我测血糖", {"need_reminder"}),
        ("我今天早上血压145/92，帮我记录一下，这正常吗？", {"need_record", "need_advice"}),
    ],
)
def test_real_model_routing(runtime, text, expected_flags):
    result = runtime.predict(text)
    payload = result.model_dump()

    assert set(payload) == {
        "routing",
        "confidence",
        "slots",
        "missing_fields",
        "risk_flags",
        "normalized",
    }
    for flag in expected_flags:
        assert payload["routing"][flag] is True
    if "need_reminder" in expected_flags:
        reminder = payload["slots"]["reminder"]
        assert "repeat" in reminder
        assert "recurrence" not in reminder
        if reminder.get("time"):
            assert "reminder.time" not in payload["missing_fields"]
