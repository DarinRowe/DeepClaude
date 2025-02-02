"""Claude API 客户端"""
import json
from typing import AsyncGenerator
from app.utils.logger import logger
from .base_client import BaseClient


class ClaudeClient(BaseClient):
    def __init__(self, api_key: str, api_url: str = "https://aihubmix.com/v1/messages"):
        """初始化 Claude 客户端
        
        Args:
            api_key: Claude API密钥
            api_url: Claude API地址
        """
        super().__init__(api_key, api_url)
        
    async def stream_chat(self, messages: list, model: str = "claude-3-5-sonnet-20241022") -> AsyncGenerator[tuple[str, str], None]:
        """流式对话
        
        Args:
            messages: 消息列表
            model: 模型名称
            
        Yields:
            tuple[str, str]: (内容类型, 内容)
                内容类型: "answer"
                内容: 实际的文本内容
        """
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "accept": "text/event-stream",
        }
        
        data = {
            "model": model,
            "messages": messages,
            "max_tokens": 8192,
            "stream": True
        }
        
        async for chunk in self._make_request(headers, data):
            chunk_str = chunk.decode('utf-8')
            if not chunk_str.strip():
                continue
                
            for line in chunk_str.split('\n'):
                if line.startswith('data: '):
                    json_str = line[6:]  # 去掉 'data: ' 前缀
                    if json_str.strip() == '[DONE]':
                        return
                        
                    try:
                        data = json.loads(json_str)
                        if data.get('type') == 'content_block_delta':
                            content = data.get('delta', {}).get('text', '')
                            if content:
                                yield "answer", content
                    except json.JSONDecodeError:
                        continue
