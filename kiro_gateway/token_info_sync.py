# -*- coding: utf-8 -*-

"""
Token 信息同步服务。

后台定期同步 Token 的账户信息（邮箱、订阅、使用量等）。
"""

import asyncio
import time
from typing import Optional
from loguru import logger

from kiro_gateway.database import UserDatabase
from kiro_gateway.auth import KiroAuthManager
from kiro_gateway.kiro_api import fetch_token_info


# 全局数据库实例
user_db = UserDatabase()


class TokenInfoSyncService:
    """
    Token 信息同步服务。
    
    定期从 Kiro API 获取账户信息并更新到数据库。
    """
    
    def __init__(self, sync_interval: int = 1800):
        """
        初始化同步服务。
        
        Args:
            sync_interval: 同步间隔（秒），默认 30 分钟
        """
        self.sync_interval = sync_interval
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    async def sync_token_info(self, token_id: int) -> bool:
        """
        同步单个 Token 的账户信息。
        
        Args:
            token_id: Token ID
            
        Returns:
            True if sync successful
        """
        try:
            # 1. 获取 token 凭证
            credentials = user_db.get_token_credentials(token_id)
            if not credentials or not credentials.get('refresh_token'):
                logger.warning(f"[TokenSync] Token {token_id}: No credentials found")
                return False
            
            refresh_token = credentials['refresh_token']
            client_id = credentials.get('client_id')
            client_secret = credentials.get('client_secret')
            
            # 2. 创建 AuthManager 并获取 access_token
            auth_manager = KiroAuthManager(
                refresh_token=refresh_token,
                client_id=client_id,
                client_secret=client_secret
            )
            
            access_token = await auth_manager.get_access_token()
            if not access_token:
                logger.warning(f"[TokenSync] Token {token_id}: Failed to get access token")
                return False
            
            # 3. 调用 Kiro API 获取账户信息
            # 检测 idp 类型
            idp = "BuilderId"
            if client_id and "github" in client_id.lower():
                idp = "GitHub"
            elif client_id and "google" in client_id.lower():
                idp = "Google"
            
            info = await fetch_token_info(access_token, idp)
            if not info:
                logger.warning(f"[TokenSync] Token {token_id}: Failed to fetch token info")
                return False
            
            # 4. 更新数据库
            user_db.update_token_info(
                token_id=token_id,
                email=info.get("email"),
                idp=info.get("idp", idp),
                subscription_type=info.get("subscription_type"),
                subscription_title=info.get("subscription_title"),
                usage_current=info.get("usage_current"),
                usage_limit=info.get("usage_limit"),
                base_current=info.get("base_current"),
                base_limit=info.get("base_limit"),
                trial_current=info.get("trial_current"),
                trial_limit=info.get("trial_limit"),
                trial_expiry=info.get("trial_expiry"),
                next_reset=info.get("next_reset"),
                days_remaining=info.get("days_remaining")
            )
            
            logger.info(f"[TokenSync] Token {token_id}: Synced successfully - {info.get('email', 'unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"[TokenSync] Token {token_id}: Sync failed - {e}")
            return False
    
    async def sync_all_tokens(self) -> dict:
        """
        同步所有活跃 Token 的账户信息。
        
        Returns:
            {"success": N, "failed": M, "total": N+M}
        """
        tokens = user_db.get_all_active_tokens()
        success = 0
        failed = 0
        
        logger.info(f"[TokenSync] Starting sync for {len(tokens)} active tokens...")
        
        for token in tokens:
            try:
                if await self.sync_token_info(token.id):
                    success += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"[TokenSync] Token {token.id}: {e}")
                failed += 1
            
            # 避免请求过快
            await asyncio.sleep(0.5)
        
        logger.info(f"[TokenSync] Sync completed: {success} success, {failed} failed")
        return {"success": success, "failed": failed, "total": success + failed}
    
    async def _background_sync_loop(self):
        """后台同步循环。"""
        logger.info(f"[TokenSync] Background sync started (interval: {self.sync_interval}s)")
        
        while self._running:
            try:
                # 等待指定间隔
                await asyncio.sleep(self.sync_interval)
                
                if self._running:
                    await self.sync_all_tokens()
                    
            except asyncio.CancelledError:
                logger.info("[TokenSync] Background sync cancelled")
                break
            except Exception as e:
                logger.error(f"[TokenSync] Background sync error: {e}")
    
    def start(self):
        """启动后台同步服务。"""
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._background_sync_loop())
        logger.info("[TokenSync] Token info sync service started")
    
    def stop(self):
        """停止后台同步服务。"""
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        logger.info("[TokenSync] Token info sync service stopped")


# 全局同步服务实例
token_sync_service = TokenInfoSyncService()
