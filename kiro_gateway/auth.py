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
Kiro API Authentication Manager.

Manages access token lifecycle:
- Load credentials from .env or JSON file
- Auto-refresh token on expiration
- Thread-safe refresh using asyncio.Lock
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
from loguru import logger

from kiro_gateway.config import (
    TOKEN_REFRESH_THRESHOLD,
    get_kiro_refresh_url,
    get_kiro_api_host,
    get_kiro_q_host,
)
from kiro_gateway.utils import get_machine_fingerprint


class KiroAuthManager:
    """
    Manages token lifecycle for Kiro API access.

    Supports:
    - Loading credentials from .env or JSON file
    - Auto-refresh token on expiration
    - Checking expiration time (expiresAt)
    - Saving updated tokens to file

    Attributes:
        profile_arn: AWS CodeWhisperer profile ARN
        region: AWS region
        api_host: API host for current region
        q_host: Q API host for current region
        fingerprint: Unique machine fingerprint

    Example:
        >>> auth_manager = KiroAuthManager(
        ...     refresh_token="your_refresh_token",
        ...     region="us-east-1"
        ... )
        >>> token = await auth_manager.get_access_token()
    """

    def __init__(
        self,
        refresh_token: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        profile_arn: Optional[str] = None,
        region: str = "us-east-1",
        creds_file: Optional[str] = None
    ):
        """
        Initialize authentication manager.

        Args:
            refresh_token: Refresh token for obtaining access token
            client_id: OIDC client ID (for Kiro Account Manager tokens)
            client_secret: OIDC client secret (for Kiro Account Manager tokens)
            profile_arn: AWS CodeWhisperer profile ARN
            region: AWS region (default us-east-1)
            creds_file: Path to JSON credentials file (optional)
        """
        self._refresh_token = refresh_token
        self._client_id = client_id
        self._client_secret = client_secret
        self._profile_arn = profile_arn
        self._region = region
        self._creds_file = creds_file

        self._access_token: Optional[str] = None
        self._expires_at: Optional[datetime] = None
        self._lock = asyncio.Lock()

        # Dynamic URLs based on region
        self._refresh_url = get_kiro_refresh_url(region)
        self._api_host = get_kiro_api_host(region)
        self._q_host = get_kiro_q_host(region)

        # Fingerprint for User-Agent
        self._fingerprint = get_machine_fingerprint()

        # Load credentials from file if specified
        if creds_file:
            self._load_credentials_from_file(creds_file)

    @staticmethod
    def _is_url(path: str) -> bool:
        """Check if path is a URL."""
        return path.startswith(('http://', 'https://'))

    def _load_credentials_from_file(self, file_path: str) -> None:
        """
        Load credentials from JSON file or remote URL.

        Supported fields in JSON:
        - refreshToken: Refresh token
        - accessToken: Access token (if already available)
        - profileArn: Profile ARN
        - region: AWS region
        - expiresAt: Token expiration time (ISO 8601)

        Args:
            file_path: Path to JSON file or remote URL (http/https)
        """
        try:
            if self._is_url(file_path):
                # Fetch from remote URL
                response = httpx.get(file_path, timeout=10.0, follow_redirects=True)
                response.raise_for_status()
                data = response.json()
                logger.info(f"Credentials loaded from URL: {file_path}")
            else:
                # Load from local file
                path = Path(file_path).expanduser()
                if not path.exists():
                    logger.warning(f"Credentials file not found: {file_path}")
                    return

                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.info(f"Credentials loaded from file: {file_path}")

            # Handle list of credentials (e.g. from Kiro Account Manager)
            # Case 1: Root list
            if isinstance(data, list) and len(data) > 0:
                logger.info(f"Credentials file contains {len(data)} accounts (list), using the first one.")
                data = data[0]
            elif isinstance(data, list) and len(data) == 0:
                logger.warning("Credentials list is empty")
                return
            
            # Case 2: Object with "accounts" list (Kiro Account Manager export)
            if isinstance(data, dict) and 'accounts' in data and isinstance(data['accounts'], list):
                 if len(data['accounts']) > 0:
                     logger.info(f"Credentials file contains {len(data['accounts'])} accounts (export object), using the first one.")
                     data = data['accounts'][0]
                 else:
                     logger.warning("Credentials 'accounts' list is empty")
                     return

                     logger.warning("Credentials 'accounts' list is empty")
                     return
            
            logger.info(f"Loaded account keys: {list(data.keys())}")
            
            # Helper to extract from nested credentials
            if 'credentials' in data and isinstance(data['credentials'], dict):
                data = data['credentials']  # Use nested credentials as data source
            
            if 'refreshToken' in data:
                self._refresh_token = data['refreshToken']
            else:
                logger.warning("refreshToken key NOT found in account data!")

            if 'accessToken' in data:
                self._access_token = data['accessToken']
            if 'clientId' in data:
                self._client_id = data['clientId']
            if 'clientSecret' in data:
                self._client_secret = data['clientSecret']
            if 'profileArn' in data:
                self._profile_arn = data['profileArn']
            if 'region' in data:
                self._region = data['region']
                # Update URLs for new region
                self._refresh_url = get_kiro_refresh_url(self._region)
                self._api_host = get_kiro_api_host(self._region)
                self._q_host = get_kiro_q_host(self._region)

            # Parse expiresAt - support both ISO string and Unix timestamp (ms)
            if 'expiresAt' in data:
                try:
                    expires_val = data['expiresAt']
                    if isinstance(expires_val, (int, float)):
                        # Unix timestamp in milliseconds
                        if expires_val > 1e12:  # Likely milliseconds
                            expires_val = expires_val / 1000
                        self._expires_at = datetime.fromtimestamp(expires_val, tz=timezone.utc)
                    elif isinstance(expires_val, str):
                        if expires_val.endswith('Z'):
                            self._expires_at = datetime.fromisoformat(expires_val.replace('Z', '+00:00'))
                        else:
                            self._expires_at = datetime.fromisoformat(expires_val)
                except Exception as e:
                    logger.warning(f"Failed to parse expiresAt: {e}")

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error loading credentials from URL: {e}")
        except httpx.RequestError as e:
            logger.error(f"Request error loading credentials from URL: {e}")
        except Exception as e:
            logger.error(f"Error loading credentials: {e}")

    def _save_credentials_to_file(
        self,
        access_token: Optional[str] = None,
        refresh_token: Optional[str] = None,
        profile_arn: Optional[str] = None
    ) -> None:
        """
        Save updated credentials to JSON file.

        Updates existing file, preserving other fields.

        Args:
            access_token: New access token (uses current if None)
            refresh_token: New refresh token (uses current if None)
            profile_arn: New profile ARN (uses current if None)
        """
        if not self._creds_file:
            return

        try:
            path = Path(self._creds_file).expanduser()

            # Read existing data
            existing_data = {}
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)

            # Update data with provided values or current values
            existing_data['accessToken'] = access_token if access_token is not None else self._access_token
            existing_data['refreshToken'] = refresh_token if refresh_token is not None else self._refresh_token
            if self._expires_at:
                existing_data['expiresAt'] = self._expires_at.isoformat()
            if profile_arn is not None:
                existing_data['profileArn'] = profile_arn
            elif self._profile_arn:
                existing_data['profileArn'] = self._profile_arn

            # Save
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)

            logger.debug(f"Credentials saved to {self._creds_file}")

        except Exception as e:
            logger.error(f"Error saving credentials: {e}")

    def is_token_expiring_soon(self) -> bool:
        """
        Check if token is expiring soon.

        Returns:
            True if token expires within TOKEN_REFRESH_THRESHOLD seconds
            or if expiration info is missing
        """
        if not self._expires_at:
            return True

        now = datetime.now(timezone.utc)
        threshold = now.timestamp() + TOKEN_REFRESH_THRESHOLD

        return self._expires_at.timestamp() <= threshold

    async def _refresh_token_request(self) -> None:
        """
        Execute token refresh request with exponential backoff retry.

        Sends POST request to Kiro API to obtain new access token.
        Uses AWS SSO OIDC endpoint when clientId/clientSecret are present (Kiro Account Manager tokens).
        Updates internal state and saves credentials to file.

        Raises:
            ValueError: If refresh token is not set or response lacks accessToken
            httpx.HTTPError: On HTTP request error after all retries
        """
        if not self._refresh_token:
            raise ValueError("Refresh token is not set")

        logger.info("Refreshing Kiro token...")

        # Determine which endpoint and format to use based on clientId/clientSecret presence
        if self._client_id and self._client_secret:
            # AWS SSO OIDC endpoint (for Kiro Account Manager tokens)
            # Uses camelCase parameters and JSON body
            refresh_url = f"https://oidc.{self._region}.amazonaws.com/token"
            payload = {
                'grantType': 'refresh_token',
                'clientId': self._client_id,
                'clientSecret': self._client_secret,
                'refreshToken': self._refresh_token,
            }
            headers = {
                "Content-Type": "application/json",
                "User-Agent": f"KiroGateway-{self._fingerprint[:16]}",
            }
            use_oidc = True
            logger.debug(f"Using AWS SSO OIDC endpoint: {refresh_url}")
        else:
            # Kiro Desktop Auth endpoint (original behavior)
            refresh_url = self._refresh_url
            payload = {'refreshToken': self._refresh_token}
            headers = {
                "Content-Type": "application/json",
                "User-Agent": f"KiroGateway-{self._fingerprint[:16]}",
            }
            use_oidc = False
            logger.debug(f"Using Kiro Desktop endpoint: {refresh_url}")

        # 指数退避重试配置
        max_retries = 3
        base_delay = 1.0  # 初始延迟1秒
        last_error = None

        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    # Both endpoints use JSON
                    response = await client.post(refresh_url, json=payload, headers=headers)
                    response.raise_for_status()
                    data = response.json()
                break  # 成功，退出重试循环
            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code in (429, 500, 502, 503, 504):
                    # 可重试的错误
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        f"Token refresh failed (attempt {attempt + 1}/{max_retries}): "
                        f"HTTP {e.response.status_code}, retrying in {delay}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    # 不可重试的客户端错误 (4xx except 429)
                    raise
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_error = e
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    f"Token refresh failed (attempt {attempt + 1}/{max_retries}): "
                    f"{type(e).__name__}, retrying in {delay}s"
                )
                await asyncio.sleep(delay)
        else:
            # 所有重试都失败
            logger.error(f"Token refresh failed after {max_retries} attempts")
            raise last_error

        # Parse response - handle both AWS OIDC (snake_case) and Kiro Desktop (camelCase)
        new_access_token = data.get("accessToken") or data.get("access_token")
        new_refresh_token = data.get("refreshToken") or data.get("refresh_token")
        expires_in = data.get("expiresIn") or data.get("expires_in", 3600)
        new_profile_arn = data.get("profileArn")

        if not new_access_token:
            raise ValueError(f"Response does not contain accessToken: {data}")

        # Calculate expiration time with buffer (minus 60 seconds)
        now = datetime.now(timezone.utc).replace(microsecond=0)
        new_expires_at = datetime.fromtimestamp(
            now.timestamp() + expires_in - 60,
            tz=timezone.utc
        )

        # Save to file first (before updating state)
        self._save_credentials_to_file(new_access_token, new_refresh_token, new_profile_arn)

        # Update all state atomically after all operations succeed
        self._access_token = new_access_token
        if new_refresh_token:
            self._refresh_token = new_refresh_token
        if new_profile_arn:
            self._profile_arn = new_profile_arn
        self._expires_at = new_expires_at

        logger.info(f"Token refreshed, expires: {self._expires_at.isoformat()}")

    async def get_access_token(self) -> str:
        """
        Return valid access_token, refreshing if necessary.

        Thread-safe method using asyncio.Lock.
        Automatically refreshes token if expired or expiring soon.

        Returns:
            Valid access token

        Raises:
            ValueError: If unable to obtain access token
        """
        async with self._lock:
            if not self._access_token or self.is_token_expiring_soon():
                await self._refresh_token_request()

            if not self._access_token:
                raise ValueError("Failed to obtain access token")

            return self._access_token

    async def force_refresh(self) -> str:
        """
        Force token refresh.

        Used when receiving 403 error from API.

        Returns:
            New access token
        """
        async with self._lock:
            await self._refresh_token_request()
            return self._access_token

    @property
    def profile_arn(self) -> Optional[str]:
        """AWS CodeWhisperer profile ARN."""
        return self._profile_arn

    @property
    def region(self) -> str:
        """AWS region."""
        return self._region

    @property
    def api_host(self) -> str:
        """API host for current region."""
        return self._api_host

    @property
    def q_host(self) -> str:
        """Q API host for current region."""
        return self._q_host

    @property
    def fingerprint(self) -> str:
        """Unique machine fingerprint."""
        return self._fingerprint
