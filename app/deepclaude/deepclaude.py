"""DeepClaude 服务，用于协调 DeepSeek 和 Claude API 的调用"""
import json
import time
import asyncio
from typing import AsyncGenerator, Dict, Any
from app.utils.logger import logger
from app.clients import DeepSeekClient, ClaudeClient


class DeepClaude:
    """处理 DeepSeek 和 Claude API 的流式输出衔接"""
    
    __slots__ = ('deepseek_client', 'claude_client')
    
    def __init__(self, deepseek_api_key: str, claude_api_key: str, 
                 deepseek_api_url: str = "https://api.deepseek.com/v1/chat/completions", 
                 claude_api_url: str = "https://api.anthropic.com/v1/messages"):
        """初始化 API 客户端
        
        Args:
            deepseek_api_key: DeepSeek API密钥
            claude_api_key: Claude API密钥
            deepseek_api_url: DeepSeek API地址
            claude_api_url: Claude API地址
        """
        self.deepseek_client = DeepSeekClient(deepseek_api_key, deepseek_api_url)
        self.claude_client = ClaudeClient(claude_api_key, claude_api_url)
    
    async def chat_completions_with_stream(self, 
                                         messages: list[Dict[str, Any]], 
                                         deepseek_model: str = "deepseek-reasoner",
                                         claude_model: str = "claude-3-5-sonnet-20241022",
                                         timeout: float = 300.0) -> AsyncGenerator[bytes, None]:
        """处理完整的流式输出过程
        
        Args:
            messages: 初始消息列表
            deepseek_model: DeepSeek 模型名称
            claude_model: Claude 模型名称
            timeout: 整体超时时间（秒）
            
        Yields:
            字节流数据
        """
        chat_id = f"chatcmpl-{hex(int(time.time() * 1000))[2:]}"
        created_time = int(time.time())

        output_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        claude_queue: asyncio.Queue[str] = asyncio.Queue()
        reasoning_content: list[str] = []
        
        async def process_deepseek():
            try:
                async for content_type, content in self.deepseek_client.stream_chat(messages, deepseek_model):
                    if not content:  # 跳过空内容
                        continue
                        
                    if content_type == "reasoning":
                        reasoning_content.append(content)
                        response = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": deepseek_model,
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "role": "assistant",
                                    "reasoning_content": content,
                                    "content": content
                                }
                            }]
                        }
                        await output_queue.put(f"data: {json.dumps(response)}\n\n".encode('utf-8'))
                    elif content_type == "content":
                        if not reasoning_content:
                            logger.warning("DeepSeek 未返回推理内容就结束了流")
                        await claude_queue.put("".join(reasoning_content))
                        break
            except asyncio.CancelledError:
                logger.info("DeepSeek 任务被取消")
                raise
            except Exception as e:
                logger.error(f"处理 DeepSeek 流时发生错误: {str(e)}", exc_info=True)
                await claude_queue.put("")
            finally:
                await output_queue.put(None)
        
        async def process_claude():
            try:
                reasoning = await claude_queue.get()
                if not reasoning:
                    logger.error("未能获取到有效的推理内容，将直接使用 Claude 处理原始消息")
                    claude_messages = [message for message in messages if message.get("role", "") != "system"]
                else:
                    claude_messages = messages.copy()
                    claude_messages.append({
                        "role": "assistant",
                        "content": f"Here's my reasoning process:\n{reasoning}\n\nBased on this reasoning, I will now provide my response:"
                    })
                    claude_messages = [message for message in claude_messages if message.get("role", "") != "system"]

                async for content_type, content in self.claude_client.stream_chat(claude_messages, claude_model):
                    if not content:  # 跳过空内容
                        continue
                        
                    if content_type == "answer":
                        response = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": claude_model,
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "role": "assistant",
                                    "content": content
                                }
                            }]
                        }
                        await output_queue.put(f"data: {json.dumps(response)}\n\n".encode('utf-8'))
            except asyncio.CancelledError:
                logger.info("Claude 任务被取消")
                raise
            except Exception as e:
                logger.error(f"处理 Claude 流时发生错误: {str(e)}", exc_info=True)
            finally:
                await output_queue.put(None)
        
        try:
            deepseek_task = asyncio.create_task(process_deepseek())
            claude_task = asyncio.create_task(process_claude())
            
            finished_tasks = 0
            async with asyncio.timeout(timeout):
                while finished_tasks < 2:
                    item = await output_queue.get()
                    if item is None:
                        finished_tasks += 1
                    else:
                        yield item
                
                yield b'data: [DONE]\n\n'
                
        except asyncio.TimeoutError:
            logger.error(f"处理超时，已超过 {timeout} 秒")
            deepseek_task.cancel()
            claude_task.cancel()
            yield b'data: {"error": "Operation timeout"}\n\n'
            yield b'data: [DONE]\n\n'
        finally:
            # 确保任务被清理
            for task in [deepseek_task, claude_task]:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass