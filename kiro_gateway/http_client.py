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
HTTP 客户端管理器。

全局 HTTP 客户端连接池管理，提高性能。
"""

import asyncio
from typing import Optional

import httpx
from loguru import logger

from kiro_gateway.auth import KiroAuthManager
from kiro_gateway.config import settings


class GlobalHTTPClientManager:
    """
    全局 HTTP 客户端管理器。

    维护全局连接池，避免为每个请求创建新的客户端。
    支持配置连接池参数。
    """

    def __init__(self):
        """初始化全局客户端管理器。"""
        self._client: Optional[httpx.AsyncClient] = None
        self._lock = asyncio.Lock()

    async def get_client(self, timeout: float = 300) -> httpx.AsyncClient:
        """
        获取或创建 HTTP 客户端。

        Args:
            timeout: 请求超时时间（秒）

        Returns:
            HTTP 客户端实例
        """
        async with self._lock:
            if self._client is None or self._client.is_closed:
                # 配置连接池参数
                limits = httpx.Limits(
                    max_connections=100,  # 最大连接数
                    max_keepalive_connections=20,  # 最大保持活动连接数
                    keepalive_expiry=30.0  # 保持活动连接的过期时间
                )

                self._client = httpx.AsyncClient(
                    timeout=timeout,
                    follow_redirects=True,
                    limits=limits,
                    http2=False  # HTTP/2 需要额外安装 httpx[http2]
                )
                logger.debug("Created new global HTTP client with connection pool")

            return self._client

    async def close(self) -> None:
        """关闭全局 HTTP 客户端。"""
        async with self._lock:
            if self._client and not self._client.is_closed:
                await self._client.aclose()
                logger.debug("Closed global HTTP client")


# 创建全局管理器实例
global_http_client_manager = GlobalHTTPClientManager()


class KiroHttpClient:
    """
    Kiro API HTTP 客户端，支持重试逻辑。

    使用全局连接池以提高性能。
    自动处理各种错误类型：
    - 403: 自动刷新 token 并重试
    - 429: 指数退避重试
    - 5xx: 指数退避重试
    - 超时: 指数退避重试
    """

    def __init__(self, auth_manager: KiroAuthManager):
        """
        初始化 HTTP 客户端。

        Args:
            auth_manager: 认证管理器
        """
        self.auth_manager = auth_manager
        self.client = None  # 将使用全局客户端

    async def _get_client(self, timeout: float = 300) -> httpx.AsyncClient:
        """
        获取 HTTP 客户端（使用全局连接池）。

        Args:
            timeout: 请求超时时间（秒）

        Returns:
            HTTP 客户端实例
        """
        return await global_http_client_manager.get_client(timeout)

    async def close(self) -> None:
        """
        关闭客户端（实际上不关闭全局客户端）。

        保留此方法以保持向后兼容性。
        """
        pass

    async def request_with_retry(
        self,
        method: str,
        url: str,
        json_data: dict,
        stream: bool = False,
        first_token_timeout: float = None
    ) -> httpx.Response:
        """
        执行带重试逻辑的 HTTP 请求。

        自动处理各种错误类型：
        - 403: 刷新 token 并重试
        - 429: 指数退避重试
        - 5xx: 指数退避重试
        - 超时: 指数退避重试

        Args:
            method: HTTP 方法
            url: 请求 URL
            json_data: JSON 请求体
            stream: 是否使用流式响应
            first_token_timeout: 首个 token 超时时间（仅用于流式）

        Returns:
            HTTP 响应

        Raises:
            HTTPException: 重试失败后抛出
        """
        # 根据是否为流式设置超时和重试次数
        if stream:
            timeout = first_token_timeout or settings.first_token_timeout
            max_retries = settings.first_token_max_retries
        else:
            timeout = 300
            max_retries = settings.max_retries

        client = await self._get_client(timeout)
        last_error = None

        for attempt in range(max_retries):
            try:
                # 获取当前有效的 token
                token = await self.auth_manager.get_access_token()
                headers = self._get_headers(token)

                if stream:
                    req = client.build_request(method, url, json=json_data, headers=headers)
                    response = await client.send(req, stream=True)
                else:
                    response = await client.request(method, url, json=json_data, headers=headers)

                # 检查响应状态
                if response.status_code == 200:
                    return response

                # 403 - token 过期，刷新并重试
                if response.status_code == 403:
                    logger.warning(f"Received 403, refreshing token (attempt {attempt + 1}/{settings.max_retries})")
                    await self.auth_manager.force_refresh()
                    continue

                # 429 - 速率限制，等待后重试
                if response.status_code == 429:
                    delay = settings.base_retry_delay * (2 ** attempt)
                    logger.warning(f"Received 429, waiting {delay}s (attempt {attempt + 1}/{settings.max_retries})")
                    await asyncio.sleep(delay)
                    continue

                # 5xx - 服务器错误，等待后重试
                if 500 <= response.status_code < 600:
                    delay = settings.base_retry_delay * (2 ** attempt)
                    logger.warning(f"Received {response.status_code}, waiting {delay}s (attempt {attempt + 1}/{settings.max_retries})")
                    await asyncio.sleep(delay)
                    continue

                # 其他错误直接返回
                return response

            except httpx.TimeoutException as e:
                last_error = e
                if stream:
                    # 流式请求的首个 token 超时
                    logger.warning(f"First token timeout after {timeout}s (attempt {attempt + 1}/{max_retries})")
                else:
                    delay = settings.base_retry_delay * (2 ** attempt)
                    logger.warning(f"Timeout, waiting {delay}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)

            except httpx.RequestError as e:
                last_error = e
                delay = settings.base_retry_delay * (2 ** attempt)
                logger.warning(f"Request error: {e}, waiting {delay}s (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(delay)

        # 所有重试都失败
        if stream:
            raise HTTPException(
                status_code=504,
                detail=f"Model did not respond within {timeout}s after {max_retries} attempts. Please try again."
            )
        else:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to complete request after {max_retries} attempts: {last_error}"
            )

    def _get_headers(self, token: str) -> dict:
        """
        构建请求头。

        Args:
            token: 访问令牌

        Returns:
            请求头字典
        """
        from kiro_gateway.utils import get_kiro_headers
        return get_kiro_headers(self.auth_manager, token)

    async def __aenter__(self) -> "KiroHttpClient":
        """支持 async context manager。"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """退出上下文时不关闭全局客户端。"""
        pass


async def close_global_http_client():
    """关闭全局 HTTP 客户端（应用关闭时调用）。"""
    await global_http_client_manager.close()