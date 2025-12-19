# -*- coding: utf-8 -*-

"""
KiroGate 性能测试。

使用 Locust 进行负载测试。

运行方式:
    locust -f tests/performance/locustfile.py --host=http://localhost:8000

Web UI 访问: http://localhost:8089
"""

import json
import os
from locust import HttpUser, task, between


# 从环境变量获取 API Key
API_KEY = os.getenv("PROXY_API_KEY", "changeme_proxy_secret")


class KiroGateUser(HttpUser):
    """
    模拟 KiroGate 用户的负载测试。
    """

    # 请求间隔时间（秒）
    wait_time = between(1, 3)

    def on_start(self):
        """用户启动时的初始化。"""
        self.headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        self.anthropic_headers = {
            "x-api-key": API_KEY,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }

    @task(1)
    def health_check(self):
        """健康检查端点测试。"""
        self.client.get("/health")

    @task(1)
    def get_models(self):
        """获取模型列表测试。"""
        self.client.get("/v1/models", headers=self.headers)

    @task(3)
    def chat_completions_simple(self):
        """简单聊天补全测试（非流式）。"""
        payload = {
            "model": "claude-sonnet-4",
            "messages": [
                {"role": "user", "content": "Say hello in one word."}
            ],
            "max_tokens": 10,
            "stream": False
        }
        self.client.post(
            "/v1/chat/completions",
            headers=self.headers,
            json=payload,
            name="/v1/chat/completions (non-stream)"
        )

    @task(3)
    def chat_completions_stream(self):
        """流式聊天补全测试。"""
        payload = {
            "model": "claude-sonnet-4",
            "messages": [
                {"role": "user", "content": "Count from 1 to 5."}
            ],
            "max_tokens": 50,
            "stream": True
        }
        with self.client.post(
            "/v1/chat/completions",
            headers=self.headers,
            json=payload,
            stream=True,
            name="/v1/chat/completions (stream)",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                # 消费流式响应
                for _ in response.iter_lines():
                    pass
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(2)
    def anthropic_messages_simple(self):
        """Anthropic Messages API 测试（非流式）。"""
        payload = {
            "model": "claude-sonnet-4",
            "messages": [
                {"role": "user", "content": "Say hi."}
            ],
            "max_tokens": 10,
            "stream": False
        }
        self.client.post(
            "/v1/messages",
            headers=self.anthropic_headers,
            json=payload,
            name="/v1/messages (non-stream)"
        )

    @task(2)
    def anthropic_messages_stream(self):
        """Anthropic Messages API 流式测试。"""
        payload = {
            "model": "claude-sonnet-4",
            "messages": [
                {"role": "user", "content": "Count 1 to 3."}
            ],
            "max_tokens": 30,
            "stream": True
        }
        with self.client.post(
            "/v1/messages",
            headers=self.anthropic_headers,
            json=payload,
            stream=True,
            name="/v1/messages (stream)",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                for _ in response.iter_lines():
                    pass
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")


class ToolCallUser(HttpUser):
    """
    测试 Tool Calls 功能的用户。
    """

    wait_time = between(2, 5)
    weight = 1  # 较低权重

    def on_start(self):
        """用户启动时的初始化。"""
        self.headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

    @task
    def chat_with_tools(self):
        """带工具调用的聊天测试。"""
        payload = {
            "model": "claude-sonnet-4",
            "messages": [
                {"role": "user", "content": "What's the weather in Tokyo?"}
            ],
            "max_tokens": 100,
            "stream": False,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather for a location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "City name"
                                }
                            },
                            "required": ["location"]
                        }
                    }
                }
            ]
        }
        self.client.post(
            "/v1/chat/completions",
            headers=self.headers,
            json=payload,
            name="/v1/chat/completions (with tools)"
        )
