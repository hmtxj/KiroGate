# -*- coding: utf-8 -*-

"""
Kiro API 客户端模块。

用于调用 Kiro 官方 API 获取账户信息、使用量等数据。
"""

import cbor2
import httpx
import uuid
from typing import Optional, Dict, Any
from loguru import logger


# Kiro API 基础 URL
KIRO_API_BASE = "https://app.kiro.dev/service/KiroWebPortalService/operation"


def _generate_invocation_id() -> str:
    """生成 AWS SDK 调用 ID"""
    return str(uuid.uuid4())


async def kiro_api_request(
    operation: str,
    body: Dict[str, Any],
    access_token: str,
    idp: str = "BuilderId"
) -> Dict[str, Any]:
    """
    调用 Kiro API。
    
    Args:
        operation: API 操作名称，如 "GetUserUsageAndLimits"
        body: 请求体（Python dict，会被 CBOR 编码）
        access_token: 访问令牌
        idp: 身份提供商 (BuilderId, GitHub, Google)
        
    Returns:
        API 响应（CBOR 解码后的 dict）
        
    Raises:
        httpx.HTTPError: 请求失败
        cbor2.CBORDecodeError: 响应解码失败
    """
    url = f"{KIRO_API_BASE}/{operation}"
    
    # CBOR 编码请求体
    payload = cbor2.dumps(body)
    
    headers = {
        "Accept": "application/cbor",
        "Content-Type": "application/cbor",
        "smithy-protocol": "rpc-v2-cbor",
        "amz-sdk-invocation-id": _generate_invocation_id(),
        "amz-sdk-request": "attempt=1; max=1",
        "x-amz-user-agent": "aws-sdk-js/1.0.0 kiro-gateway/1.0.0",
        "Authorization": f"Bearer {access_token}",
        "Cookie": f"Idp={idp}; AccessToken={access_token}"
    }
    
    logger.debug(f"[Kiro API] Calling {operation}")
    
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(url, content=payload, headers=headers)
        
        if not response.is_success:
            # 尝试解析 CBOR 错误响应
            error_msg = f"HTTP {response.status_code}"
            try:
                error_data = cbor2.loads(response.content)
                if isinstance(error_data, dict):
                    if "__type" in error_data and "message" in error_data:
                        error_type = error_data["__type"].split("#")[-1]
                        error_msg = f"{error_type}: {error_data['message']}"
                    elif "message" in error_data:
                        error_msg = error_data["message"]
            except Exception:
                pass
            logger.error(f"[Kiro API] {operation} failed: {error_msg}")
            raise httpx.HTTPStatusError(error_msg, request=response.request, response=response)
        
        result = cbor2.loads(response.content)
        logger.debug(f"[Kiro API] {operation} succeeded")
        return result


async def get_user_info(access_token: str, idp: str = "BuilderId") -> Dict[str, Any]:
    """
    获取用户基本信息。
    
    Returns:
        {
            "email": "user@example.com",
            "userId": "xxx",
            "idp": "BuilderId",
            "status": "Active",
            "featureFlags": [...]
        }
    """
    return await kiro_api_request(
        "GetUserInfo",
        {"origin": "KIRO_IDE"},
        access_token,
        idp
    )


async def get_user_usage_and_limits(
    access_token: str,
    idp: str = "BuilderId"
) -> Dict[str, Any]:
    """
    获取用户使用量和订阅限制信息。
    
    Returns:
        {
            "userInfo": {"email": "...", "userId": "..."},
            "subscriptionInfo": {
                "type": "FREE",
                "subscriptionTitle": "KIRO FREE",
                "upgradeCapability": "...",
                "overageCapability": "...",
                "subscriptionManagementTarget": "..."
            },
            "usageBreakdownList": [{
                "resourceType": "CREDIT",
                "currentUsage": 0,
                "currentUsageWithPrecision": 0.0,
                "usageLimit": 50,
                "usageLimitWithPrecision": 50.0,
                "displayName": "Credits",
                "displayNamePlural": "Credits",
                "freeTrialInfo": {
                    "currentUsage": 0,
                    "usageLimit": 500,
                    "freeTrialStatus": "ACTIVE",
                    "freeTrialExpiry": "2026-02-03T00:00:00Z"
                },
                "bonuses": []
            }],
            "nextDateReset": "2026-02-01T00:00:00Z",
            "overageConfiguration": {"overageEnabled": false}
        }
    """
    return await kiro_api_request(
        "GetUserUsageAndLimits",
        {"isEmailRequired": True, "origin": "KIRO_IDE"},
        access_token,
        idp
    )


def parse_usage_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    解析 GetUserUsageAndLimits 响应，提取关键信息。
    
    Returns:
        {
            "email": "user@example.com",
            "userId": "xxx",
            "subscription_type": "Free",
            "subscription_title": "KIRO FREE",
            "usage_current": 0.0,
            "usage_limit": 550.0,
            "base_current": 0.0,
            "base_limit": 50.0,
            "trial_current": 0.0,
            "trial_limit": 500.0,
            "trial_expiry": "2026-02-03",
            "next_reset": "2026-02-01",
            "days_remaining": 27,
            "idp": "BuilderId",
            "upgrade_capability": "...",
            "overage_capability": "..."
        }
    """
    result = {}
    
    # 用户信息
    user_info = data.get("userInfo", {})
    result["email"] = user_info.get("email")
    result["userId"] = user_info.get("userId")
    
    # 订阅信息
    sub_info = data.get("subscriptionInfo", {})
    sub_title = sub_info.get("subscriptionTitle", "Free")
    result["subscription_title"] = sub_title
    
    # 解析订阅类型
    sub_type = "Free"
    if "PRO" in sub_title.upper():
        sub_type = "Pro"
    elif "ENTERPRISE" in sub_title.upper():
        sub_type = "Enterprise"
    elif "TEAMS" in sub_title.upper():
        sub_type = "Teams"
    result["subscription_type"] = sub_type
    
    result["upgrade_capability"] = sub_info.get("upgradeCapability")
    result["overage_capability"] = sub_info.get("overageCapability")
    result["management_target"] = sub_info.get("subscriptionManagementTarget")
    
    # 使用量信息
    usage_list = data.get("usageBreakdownList", [])
    credit_usage = None
    for item in usage_list:
        if item.get("resourceType") == "CREDIT" or item.get("displayName") == "Credits":
            credit_usage = item
            break
    
    if credit_usage:
        # 基础额度（优先使用精确小数）
        base_limit = credit_usage.get("usageLimitWithPrecision") or credit_usage.get("usageLimit", 0)
        base_current = credit_usage.get("currentUsageWithPrecision") or credit_usage.get("currentUsage", 0)
        result["base_limit"] = base_limit
        result["base_current"] = base_current
        
        # 试用额度
        trial_info = credit_usage.get("freeTrialInfo", {})
        trial_limit = 0
        trial_current = 0
        trial_expiry = None
        if trial_info.get("freeTrialStatus") == "ACTIVE":
            trial_limit = trial_info.get("usageLimitWithPrecision") or trial_info.get("usageLimit", 0)
            trial_current = trial_info.get("currentUsageWithPrecision") or trial_info.get("currentUsage", 0)
            trial_expiry = trial_info.get("freeTrialExpiry")
        result["trial_limit"] = trial_limit
        result["trial_current"] = trial_current
        # trial_expiry 可能是 datetime 对象或字符串
        if trial_expiry:
            if hasattr(trial_expiry, 'isoformat'):
                trial_expiry = trial_expiry.isoformat()
            result["trial_expiry"] = str(trial_expiry)[:10]  # 只保留日期部分
        else:
            result["trial_expiry"] = None
        
        # 奖励额度
        bonuses = credit_usage.get("bonuses", [])
        bonus_limit = sum(
            b.get("usageLimitWithPrecision") or b.get("usageLimit", 0)
            for b in bonuses if b.get("status") == "ACTIVE"
        )
        bonus_current = sum(
            b.get("currentUsageWithPrecision") or b.get("currentUsage", 0)
            for b in bonuses if b.get("status") == "ACTIVE"
        )
        result["bonus_limit"] = bonus_limit
        result["bonus_current"] = bonus_current
        
        # 总使用量
        result["usage_limit"] = base_limit + trial_limit + bonus_limit
        result["usage_current"] = base_current + trial_current + bonus_current
    else:
        result["base_limit"] = 0
        result["base_current"] = 0
        result["trial_limit"] = 0
        result["trial_current"] = 0
        result["trial_expiry"] = None
        result["bonus_limit"] = 0
        result["bonus_current"] = 0
        result["usage_limit"] = 0
        result["usage_current"] = 0
    
    # 重置日期
    next_reset = data.get("nextDateReset")
    # next_reset 可能是 datetime 对象或字符串
    if next_reset:
        if hasattr(next_reset, 'isoformat'):
            next_reset_str = next_reset.isoformat()
        else:
            next_reset_str = str(next_reset)
        result["next_reset"] = next_reset_str[:10]  # 只保留日期部分
    else:
        result["next_reset"] = None
    
    # 计算剩余天数
    if next_reset:
        from datetime import datetime, timezone
        try:
            # next_reset 可能已经是 datetime 对象
            if hasattr(next_reset, 'tzinfo'):
                reset_date = next_reset
            else:
                next_reset_str = str(next_reset)
                reset_date = datetime.fromisoformat(next_reset_str.replace("Z", "+00:00"))
            
            if reset_date.tzinfo:
                now = datetime.now(reset_date.tzinfo)
            else:
                now = datetime.now(timezone.utc)
                reset_date = reset_date.replace(tzinfo=timezone.utc)
            
            days = (reset_date - now).days
            result["days_remaining"] = max(0, days)
        except Exception as e:
            logger.warning(f"[Kiro API] Failed to parse reset date: {e}")
            result["days_remaining"] = None
    else:
        result["days_remaining"] = None
    
    return result


async def fetch_token_info(
    access_token: str,
    idp: str = "BuilderId"
) -> Optional[Dict[str, Any]]:
    """
    获取并解析 Token 的完整账户信息。
    
    这是一个便捷函数，组合调用 API 并解析返回数据。
    
    Args:
        access_token: 访问令牌
        idp: 身份提供商
        
    Returns:
        解析后的账户信息 dict，失败返回 None
    """
    try:
        data = await get_user_usage_and_limits(access_token, idp)
        result = parse_usage_response(data)
        result["idp"] = idp
        return result
    except Exception as e:
        logger.error(f"[Kiro API] Failed to fetch token info: {e}")
        return None
