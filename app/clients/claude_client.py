"""Claude API 客户端"""
from typing import AsyncGenerator
from openai import AsyncOpenAI
from app.utils.logger import logger
from .base_client import BaseClient


class ClaudeClient(BaseClient):
    def __init__(self, api_key: str, api_url: str = "https://openrouter.ai/api/v1"):
        """初始化 Claude 客户端
        
        Args:
            api_key: OpenRouter API密钥
            api_url: OpenRouter API基础地址
        """
        super().__init__(api_key, api_url)
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_url
        )
        
    async def stream_chat(self, messages: list, model: str = "anthropic/claude-3-sonnet") -> AsyncGenerator[tuple[str, str], None]:
        """流式对话
        
        Args:
            messages: 消息列表
            model: 模型名称
            
        Yields:
            tuple[str, str]: (内容类型, 内容)
                内容类型: "answer"
                内容: 实际的文本内容
        """
        try:
            stream = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                max_tokens=8192,
                temperature=0.7,
                extra_headers={
                    "HTTP-Referer": "https://github.com/wangjueszu/chatgpt-web-share",
                    "X-Title": "chatgpt-web-share",
                }
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield "answer", chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"OpenRouter API 请求失败: {e}")
