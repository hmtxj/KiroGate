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
模型元数据缓存。

线程安全的存储信息，支持 TTL、lazy loading 和后台刷新。
"""

import asyncio
import time
from typing import Any, Dict, List, Optional

import httpx
from loguru import logger

from kiro_gateway.config import MODEL_CACHE_TTL, DEFAULT_MAX_INPUT_TOKENS


class ModelInfoCache:
    """
    线程安全的模型元数据缓存。

    使用 Lazy Loading 填充 - 仅在首次访问或缓存过期时加载数据。
    支持后台自动刷新机制。
    """

    def __init__(self, cache_ttl: int = MODEL_CACHE_TTL):
        """
        初始化模型缓存。

        Args:
            cache_ttl: 缓存 TTL（秒）（默认来自配置）
        """
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._last_update: Optional[float] = None
        self._cache_ttl = cache_ttl
        self._refresh_task: Optional[asyncio.Task] = None
        self._auth_manager = None  # 将由应用设置

    def set_auth_manager(self, auth_manager) -> None:
        """
        设置认证管理器（用于后台刷新）。

        Args:
            auth_manager: 认证管理器实例
        """
        self._auth_manager = auth_manager

    async def update(self, models_data: List[Dict[str, Any]]) -> None:
        """
        更新模型缓存。

        线程安全地替换缓存内容为新数据。

        Args:
            models_data: 模型信息字典列表
                      每个字典应包含 "modelId" 键
        """
        async with self._lock:
            logger.info(f"Updating model cache. Found {len(models_data)} models.")
            self._cache = {model["modelId"]: model for model in models_data}
            self._last_update = time.time()

    async def refresh(self) -> bool:
        """
        从 API 刷新缓存。

        Returns:
            True 如果刷新成功，False 否则
        """
        if not self._auth_manager:
            logger.warning("No auth manager set, cannot refresh cache")
            return False

        try:
            token = await self._auth_manager.get_access_token()
            from kiro_gateway.utils import get_kiro_headers
            headers = get_kiro_headers(self._auth_manager, token)

            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(
                    f"{self._auth_manager.q_host}/ListAvailableModels",
                    headers=headers,
                    params={
                        "origin": "AI_EDITOR",
                        "profileArn": self._auth_manager.profile_arn or ""
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    models_list = data.get("models", [])
                    await self.update(models_list)
                    logger.info(f"Successfully refreshed model cache with {len(models_list)} models")
                    return True
                else:
                    logger.error(f"Failed to refresh models: HTTP {response.status_code}")
                    return False

        except Exception as e:
            logger.error(f"Error refreshing model cache: {e}")
            return False

    async def start_background_refresh(self) -> None:
        """
        启动后台刷新任务。

        创建定期刷新缓存的后台任务。
        """
        if self._refresh_task and not self._refresh_task.done():
            logger.warning("Background refresh task is already running")
            return

        self._refresh_task = asyncio.create_task(self._background_refresh_loop())
        logger.info("Started background model cache refresh task")

    async def stop_background_refresh(self) -> None:
        """
        停止后台刷新任务。
        """
        if self._refresh_task and not self._refresh_task.done():
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                logger.info("Stopped background model cache refresh task")
            except Exception as e:
                logger.error(f"Error stopping refresh task: {e}")

    async def _background_refresh_loop(self) -> None:
        """
        后台刷新循环。

        定期刷新缓存，刷新间隔为 TTL 的一半。
        """
        refresh_interval = self._cache_ttl / 2
        logger.info(f"Background refresh will run every {refresh_interval} seconds")

        while True:
            try:
                await asyncio.sleep(refresh_interval)
                logger.debug("Running scheduled model cache refresh")
                await self.refresh()
            except asyncio.CancelledError:
                logger.info("Background refresh task cancelled")
                break
            except Exception as e:
                logger.error(f"Unexpected error in background refresh: {e}")

    def get(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        获取模型信息。

        Args:
            model_id: 模型 ID

        Returns:
            模型信息字典，如果未找到则返回 None
        """
        return self._cache.get(model_id)

    def get_max_input_tokens(self, model_id: str) -> int:
        """
        获取模型的 maxInputTokens。

        Args:
            model_id: 模型 ID

        Returns:
            最大输入 token 数或 DEFAULT_MAX_INPUT_TOKENS
        """
        model = self._cache.get(model_id)
        if model and model.get("tokenLimits"):
            return model["tokenLimits"].get("maxInputTokens") or DEFAULT_MAX_INPUT_TOKENS
        return DEFAULT_MAX_INPUT_TOKENS

    def is_empty(self) -> bool:
        """
        检查缓存是否为空。

        Returns:
            True 如果缓存为空
        """
        return not self._cache

    def is_stale(self) -> bool:
        """
        检查缓存是否过期。

        Returns:
            True 如果缓存过期（超过 cache_ttl 秒）
            或缓存从未更新
        """
        if not self._last_update:
            return True
        return time.time() - self._last_update > self._cache_ttl

    def get_all_model_ids(self) -> List[str]:
        """
        返回缓存中所有模型 ID 列表。

        Returns:
            模型 ID 列表
        """
        return list(self._cache.keys())

    @property
    def size(self) -> int:
        """缓存中的模型数量。"""
        return len(self._cache)

    @property
    def last_update_time(self) -> Optional[float]:
        """最后更新时间戳（秒）或 None。"""
        return self._last_update

    @property
    def is_background_refresh_running(self) -> bool:
        """检查后台刷新是否正在运行。"""
        return self._refresh_task is not None and not self._refresh_task.done()