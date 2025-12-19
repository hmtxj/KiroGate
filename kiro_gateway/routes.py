# -*- coding: utf-8 -*-

# KiroGate
# Based on kiro-openai-gateway by Jwadow (https://github.com/Jwadow/kiro-openai-gateway)
# Original Copyright (C) 2025 Jwadow
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
KiroGate FastAPI 路由。

包含所有 API 端点：
- / 和 /health: 健康检查
- /v1/models: 模型列表
- /v1/chat/completions: OpenAI 兼容的聊天补全
- /v1/messages: Anthropic 兼容的消息 API
"""

import json
import secrets
from datetime import datetime, timezone

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response, Security, Header
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from loguru import logger

from kiro_gateway.config import (
    PROXY_API_KEY,
    AVAILABLE_MODELS,
    APP_VERSION,
    RATE_LIMIT_PER_MINUTE,
)
from kiro_gateway.models import (
    OpenAIModel,
    ModelList,
    ChatCompletionRequest,
    AnthropicMessagesRequest,
)
from kiro_gateway.auth import KiroAuthManager
from kiro_gateway.cache import ModelInfoCache
from kiro_gateway.request_handler import RequestHandler
from kiro_gateway.utils import get_kiro_headers

# 初始化速率限制器
limiter = Limiter(key_func=get_remote_address)


def rate_limit_decorator():
    """
    条件性速率限制装饰器。

    当 RATE_LIMIT_PER_MINUTE > 0 时应用速率限制，
    当 RATE_LIMIT_PER_MINUTE = 0 时禁用速率限制。
    """
    if RATE_LIMIT_PER_MINUTE > 0:
        return limiter.limit(f"{RATE_LIMIT_PER_MINUTE}/minute")
    else:
        # 返回空装饰器（不应用速率限制）
        return lambda func: func

# 导入 debug_logger
try:
    from kiro_gateway.debug_logger import debug_logger
except ImportError:
    debug_logger = None


# --- 安全方案 ---
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


async def verify_api_key(auth_header: str = Security(api_key_header)) -> bool:
    """
    验证 Authorization header 中的 API 密钥。

    期望格式: "Bearer {PROXY_API_KEY}"

    Args:
        auth_header: Authorization header 的值

    Returns:
        True 如果密钥有效

    Raises:
        HTTPException: 401 如果密钥无效或缺失
    """
    if not auth_header or not secrets.compare_digest(auth_header, f"Bearer {PROXY_API_KEY}"):
        logger.warning("Access attempt with invalid API key.")
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return True


async def verify_anthropic_api_key(
    x_api_key: str = Header(None, alias="x-api-key"),
    auth_header: str = Security(api_key_header)
) -> bool:
    """
    验证 Anthropic 或 OpenAI 格式的 API 密钥。

    Anthropic 使用 x-api-key header，但我们也支持
    标准的 Authorization: Bearer 格式以保持兼容性。

    Args:
        x_api_key: x-api-key header 的值（Anthropic 格式）
        auth_header: Authorization header 的值（OpenAI 格式）

    Returns:
        True 如果密钥有效

    Raises:
        HTTPException: 401 如果密钥无效或缺失
    """
    # 检查 x-api-key（Anthropic 格式）
    if x_api_key and secrets.compare_digest(x_api_key, PROXY_API_KEY):
        return True

    # 检查 Authorization: Bearer（OpenAI 格式）
    if auth_header and secrets.compare_digest(auth_header, f"Bearer {PROXY_API_KEY}"):
        return True

    logger.warning("Access attempt with invalid API key (Anthropic endpoint).")
    raise HTTPException(status_code=401, detail="Invalid or missing API Key")


# --- 路由器 ---
router = APIRouter()


@router.get("/")
async def root():
    """
    健康检查端点。

    Returns:
        应用状态和版本信息
    """
    return {
        "status": "ok",
        "message": "Kiro API Gateway is running",
        "version": APP_VERSION
    }


@router.get("/health")
async def health(request: Request):
    """
    详细的健康检查。

    Returns:
        状态、时间戳、版本和运行时信息
    """
    from kiro_gateway.metrics import metrics

    auth_manager: KiroAuthManager = request.app.state.auth_manager
    model_cache: ModelInfoCache = request.app.state.model_cache

    # 检查 token 是否有效
    token_valid = False
    try:
        # 非阻塞检查 token
        if auth_manager._access_token and not auth_manager.is_token_expiring_soon():
            token_valid = True
    except Exception:
        token_valid = False

    # 更新指标
    metrics.set_cache_size(model_cache.size)
    metrics.set_token_valid(token_valid)

    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": APP_VERSION,
        "token_valid": token_valid,
        "cache_size": model_cache.size,
        "cache_last_update": model_cache.last_update_time
    }


@router.get("/metrics")
async def get_metrics():
    """
    获取 JSON 格式的应用指标。

    Returns:
        指标数据字典
    """
    from kiro_gateway.metrics import metrics
    return metrics.get_metrics()


@router.get("/metrics/prometheus")
async def get_prometheus_metrics():
    """
    获取 Prometheus 格式的应用指标。

    Returns:
        Prometheus 文本格式的指标
    """
    from kiro_gateway.metrics import metrics
    return Response(
        content=metrics.export_prometheus(),
        media_type="text/plain; charset=utf-8"
    )


@router.get("/v1/models", response_model=ModelList, dependencies=[Depends(verify_api_key)])
@rate_limit_decorator()
async def get_models(request: Request):
    """
    返回可用模型列表。

    使用静态模型列表，支持从 API 动态更新。
    缓存结果以减少 API 负载。

    Args:
        request: FastAPI Request 用于访问 app.state

    Returns:
        ModelList 包含可用模型
    """
    logger.info("Request to /v1/models")

    auth_manager: KiroAuthManager = request.app.state.auth_manager
    model_cache: ModelInfoCache = request.app.state.model_cache

    # 如果缓存为空或过期，尝试从 API 获取模型
    if model_cache.is_empty() or model_cache.is_stale():
        try:
            token = await auth_manager.get_access_token()
            headers = get_kiro_headers(auth_manager, token)

            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(
                    f"{auth_manager.q_host}/ListAvailableModels",
                    headers=headers,
                    params={
                        "origin": "AI_EDITOR",
                        "profileArn": auth_manager.profile_arn or ""
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    models_list = data.get("models", [])
                    await model_cache.update(models_list)
                    logger.info(f"Received {len(models_list)} models from API")
        except Exception as e:
            logger.warning(f"Failed to fetch models from API: {e}")

    # 返回静态模型列表
    openai_models = [
        OpenAIModel(
            id=model_id,
            owned_by="anthropic",
            description="Claude model via Kiro API"
        )
        for model_id in AVAILABLE_MODELS
    ]

    return ModelList(data=openai_models)


@router.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
@rate_limit_decorator()
async def chat_completions(request: Request, request_data: ChatCompletionRequest):
    """
    Chat completions 端点 - 兼容 OpenAI API。

    接受 OpenAI 格式的请求并转换为 Kiro API。
    支持流式和非流式模式。

    Args:
        request: FastAPI Request 用于访问 app.state
        request_data: OpenAI ChatCompletionRequest 格式的请求

    Returns:
        StreamingResponse 用于流式模式
        JSONResponse 用于非流式模式

    Raises:
        HTTPException: 验证错误或 API 错误时
    """
    logger.info(f"Request to /v1/chat/completions (model={request_data.model}, stream={request_data.stream})")

    return await RequestHandler.process_request(
        request,
        request_data,
        "/v1/chat/completions",
        convert_to_openai=False,
        response_format="openai"
    )


# ==================================================================================================
# Anthropic Messages API Endpoint (/v1/messages)
# ==================================================================================================

@router.post("/v1/messages", dependencies=[Depends(verify_anthropic_api_key)])
@rate_limit_decorator()
async def anthropic_messages(request: Request, request_data: AnthropicMessagesRequest):
    """
    Anthropic Messages API 端点 - 兼容 Anthropic SDK。

    接受 Anthropic 格式的请求并转换为 Kiro API。
    支持流式和非流式模式。

    Args:
        request: FastAPI Request 用于访问 app.state
        request_data: Anthropic MessagesRequest 格式的请求

    Returns:
        StreamingResponse 用于流式模式
        JSONResponse 用于非流式模式

    Raises:
        HTTPException: 验证错误或 API 错误时
    """
    logger.info(f"Request to /v1/messages (model={request_data.model}, stream={request_data.stream})")

    return await RequestHandler.process_request(
        request,
        request_data,
        "/v1/messages",
        convert_to_openai=True,
        response_format="anthropic"
    )


# --- 速率限制错误处理 ---
# 注意：异常处理器需要在 FastAPI 应用上注册，而不是在路由器上
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """
    处理速率限制错误。
    """
    return JSONResponse(
        status_code=429,
        content={
            "error": {
                "message": "Rate limit exceeded. Please try again later.",
                "type": "rate_limit_exceeded",
                "code": 429
            }
        }
    )