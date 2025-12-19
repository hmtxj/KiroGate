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
请求追踪中间件。

为每个请求添加唯一的 ID，用于日志关联和调试。
"""

import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger


class RequestTrackingMiddleware(BaseHTTPMiddleware):
    """
    请求追踪中间件。

    为每个请求：
    - 生成唯一的请求 ID
    - 记录请求开始和结束时间
    - 计算请求处理时间
    - 在日志中添加请求 ID 上下文
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        处理请求并添加追踪信息。

        Args:
            request: HTTP 请求
            call_next: 下一个中间件或路由处理器

        Returns:
            HTTP 响应
        """
        # 从 header 获取或生成新的请求 ID
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            request_id = str(uuid.uuid4())

        # 记录请求开始时间
        start_time = time.time()

        # 将请求 ID 添加到请求状态
        request.state.request_id = request_id

        # 使用 loguru 的 context 绑定请求 ID
        with logger.contextualize(request_id=request_id):
            # 记录请求开始
            logger.info(
                f"Request started: {request.method} {request.url.path} "
                f"(query: {request.url.query})"
            )

            try:
                # 处理请求
                response = await call_next(request)

                # 计算处理时间
                process_time = time.time() - start_time

                # 添加响应头
                response.headers["X-Request-ID"] = request_id
                response.headers["X-Process-Time"] = str(round(process_time, 4))

                # 记录请求完成
                logger.info(
                    f"Request completed: {request.method} {request.url.path} "
                    f"status={response.status_code} time={process_time:.4f}s"
                )

                return response

            except Exception as e:
                # 记录请求错误
                process_time = time.time() - start_time
                logger.error(
                    f"Request failed: {request.method} {request.url.path} "
                    f"error={str(e)} time={process_time:.4f}s"
                )
                raise


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    指标收集中间件。

    收集基本的请求指标并发送到 Prometheus 指标收集器：
    - 总请求数（按端点、状态码、模型）
    - 响应时间
    - 活跃连接数
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        收集请求指标。

        Args:
            request: HTTP 请求
            call_next: 下一个中间件或路由处理器

        Returns:
            HTTP 响应
        """
        from kiro_gateway.metrics import metrics

        start_time = time.time()
        endpoint = request.url.path
        model = "unknown"

        # 增加活跃连接数
        metrics.inc_active_connections()

        try:
            response = await call_next(request)

            # 计算处理时间
            process_time = time.time() - start_time

            # 尝试从请求状态获取模型名称
            if hasattr(request.state, "model"):
                model = request.state.model

            # 记录指标
            metrics.inc_request(endpoint, response.status_code, model)
            metrics.observe_latency(endpoint, process_time)

            return response

        except Exception as e:
            # 记录错误
            process_time = time.time() - start_time
            metrics.inc_request(endpoint, 500, model)
            metrics.inc_error(type(e).__name__)
            metrics.observe_latency(endpoint, process_time)
            raise

        finally:
            # 减少活跃连接数
            metrics.dec_active_connections()


# 创建全局指标实例
metrics_middleware = MetricsMiddleware