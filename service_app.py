"""FastAPI service for the chronic-care routing LoRA model."""

from __future__ import annotations

import hmac
import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


LOGGER = logging.getLogger("medical_lora.intent_service")
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_LORA_PATH = PROJECT_ROOT / "training" / "qwen_routing_lora_final"
SYSTEM_PROMPT = "你是慢病健康管理系统的路由与信息抽取器。你只能输出合法 JSON，不要输出解释。"


def _read_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _default_base_model_path(lora_path: Path) -> str:
    config_path = lora_path / "adapter_config.json"
    if not config_path.exists():
        return "Qwen/Qwen2.5-7B-Instruct"
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "Qwen/Qwen2.5-7B-Instruct"
    return str(payload.get("base_model_name_or_path") or "Qwen/Qwen2.5-7B-Instruct")


@dataclass(frozen=True)
class ServiceSettings:
    base_model_path: str
    lora_path: Path
    api_key: str = ""
    max_new_tokens: int = 512
    load_in_4bit: bool = True

    @classmethod
    def from_env(cls) -> "ServiceSettings":
        raw_lora_path = os.getenv("ROUTING_LORA_PATH", str(DEFAULT_LORA_PATH))
        lora_path = Path(raw_lora_path).expanduser().resolve()
        base_model_path = os.getenv("BASE_MODEL_PATH") or _default_base_model_path(lora_path)
        return cls(
            base_model_path=base_model_path,
            lora_path=lora_path,
            api_key=os.getenv("INTENT_API_KEY", ""),
            max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "512")),
            load_in_4bit=_read_bool("LOAD_IN_4BIT", True),
        )


class IntentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=2000)
    user_id: str | None = Field(default=None, max_length=200)
    conversation_id: str | None = Field(default=None, max_length=200)

    @field_validator("text")
    @classmethod
    def strip_and_validate_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text 不能为空")
        return value


class RoutingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    need_record: bool
    need_advice: bool
    need_plan: bool
    need_reminder: bool
    need_history: bool


class ConfidenceResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    need_record: float = Field(ge=0.0, le=1.0)
    need_advice: float = Field(ge=0.0, le=1.0)
    need_plan: float = Field(ge=0.0, le=1.0)
    need_reminder: float = Field(ge=0.0, le=1.0)
    need_history: float = Field(ge=0.0, le=1.0)


class IntentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    routing: RoutingResult
    confidence: ConfidenceResult
    slots: dict[str, Any]
    missing_fields: list[str]
    risk_flags: list[str]
    normalized: dict[str, Any]


class ModelNotReadyError(RuntimeError):
    """The model has not completed loading."""


class InvalidModelOutputError(RuntimeError):
    """The model returned text that does not satisfy the intent schema."""


def _slot_path_has_value(slots: dict[str, Any], path: str) -> bool:
    current: Any = slots
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return False
        current = current[part]
    return current is not None and current != "" and current != [] and current != {}


def normalize_model_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize known historical schema variants before returning them to the Agent."""
    slots = payload.get("slots")
    if not isinstance(slots, dict):
        return payload

    reminder = slots.get("reminder")
    if isinstance(reminder, dict):
        if "repeat" not in reminder and "recurrence" in reminder:
            reminder["repeat"] = reminder.pop("recurrence")

    plan = slots.get("plan")
    if isinstance(plan, dict):
        for field_name in ("preferences", "constraints"):
            if plan.get(field_name) is None:
                plan[field_name] = []

    advice = slots.get("advice")
    if isinstance(advice, dict) and advice.get("related_metrics") is None:
        advice["related_metrics"] = []

    missing_fields = payload.get("missing_fields")
    if isinstance(missing_fields, list):
        payload["missing_fields"] = [
            path
            for path in missing_fields
            if not isinstance(path, str) or not _slot_path_has_value(slots, path)
        ]
    return payload


class RoutingRuntime:
    """Load the base model once and serialize GPU inference calls."""

    def __init__(self, settings: ServiceSettings):
        self.settings = settings
        self.model = None
        self.tokenizer = None
        self.device = None
        self._inference_lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def load(self) -> None:
        if self.ready:
            return
        if not self.settings.lora_path.exists():
            raise FileNotFoundError(f"LoRA adapter path does not exist: {self.settings.lora_path}")

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if self.settings.load_in_4bit and not torch.cuda.is_available():
            raise RuntimeError("4-bit inference requires an available CUDA GPU")

        LOGGER.info("Loading tokenizer and routing adapter")
        tokenizer = AutoTokenizer.from_pretrained(
            str(self.settings.lora_path),
            trust_remote_code=True,
        )

        model_kwargs: dict[str, Any] = {
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if self.settings.load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            model_kwargs["dtype"] = torch.bfloat16

        base_model = AutoModelForCausalLM.from_pretrained(
            self.settings.base_model_path,
            **model_kwargs,
        )
        model = PeftModel.from_pretrained(base_model, str(self.settings.lora_path))
        model.eval()

        self.tokenizer = tokenizer
        self.model = model
        self.device = next(model.parameters()).device
        LOGGER.info("Routing model is ready")

    def predict(self, text: str) -> IntentResponse:
        if not self.ready:
            raise ModelNotReadyError("routing model is not ready")

        import torch

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with self._inference_lock, torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.settings.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
        raw_output = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

        try:
            payload = json.loads(raw_output)
            if not isinstance(payload, dict):
                raise TypeError("top-level model output must be an object")
            payload = normalize_model_payload(payload)
            return IntentResponse.model_validate(payload)
        except (json.JSONDecodeError, TypeError, ValidationError) as exc:
            LOGGER.error("Routing model returned invalid structured output: %s", type(exc).__name__)
            raise InvalidModelOutputError("模型未返回符合 Schema 的 JSON") from exc


def create_app(
    runtime: Any | None = None,
    settings: ServiceSettings | None = None,
    *,
    load_model_on_startup: bool = True,
) -> FastAPI:
    resolved_settings = settings or ServiceSettings.from_env()
    resolved_runtime = runtime or RoutingRuntime(resolved_settings)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.routing_runtime = resolved_runtime
        application.state.service_settings = resolved_settings
        if load_model_on_startup:
            await run_in_threadpool(resolved_runtime.load)
        yield

    application = FastAPI(
        title="Medical Lora Intent Service",
        version="1.0.0",
        lifespan=lifespan,
    )

    def verify_api_key(authorization: str | None = Header(default=None)) -> None:
        expected = resolved_settings.api_key
        if not expected:
            return
        prefix = "Bearer "
        if not authorization or not authorization.startswith(prefix):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing bearer token")
        supplied = authorization[len(prefix) :]
        if not hmac.compare_digest(supplied, expected):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid bearer token")

    @application.get("/healthz")
    def healthz():
        if not resolved_runtime.ready:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"status": "loading"},
            )
        return {"status": "ok", "model": "qwen2.5-7b-routing-lora"}

    @application.post("/v1/intent/parse", response_model=IntentResponse)
    def parse_intent(
        request: IntentRequest,
        authorization: str | None = Header(default=None),
    ):
        # Call explicitly so the endpoint can use the application-specific key setting.
        verify_api_key(authorization)
        try:
            return resolved_runtime.predict(request.text)
        except ModelNotReadyError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"code": "MODEL_NOT_READY", "message": str(exc)},
            ) from exc
        except InvalidModelOutputError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail={"code": "INVALID_MODEL_OUTPUT", "message": str(exc)},
            ) from exc
        except Exception as exc:
            LOGGER.exception("Intent inference failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"code": "INFERENCE_ERROR", "message": "意图识别失败"},
            ) from exc

    return application


app = create_app()
