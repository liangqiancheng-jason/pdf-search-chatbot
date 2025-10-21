"""
命令行 DeepSeek 聊天机器人示例，展示如何用摘要方式保存对话记忆。

详细使用步骤：
    1. 运行脚本前请先准备好 DeepSeek 的 API Key，可以写进 `.env` 文件或直接导出环境变量，
       例如执行：export DEEPSEEK_API_KEY=你的密钥。
    2. 如果你有自建代理或者不同的网关地址，可以设置 `DEEPSEEK_API_BASE`，否则保持默认即可。
    3. 命令行运行：python deepseek_chatbot.py，然后按照提示输入问题。

实现思路说明：
    - 每轮对话结束后，程序会自动调用 DeepSeek 生成简短摘要，只保留关键信息；
    - 下次提问时，模型只会看到摘要和上一轮的详细问答，从而减少 Token 消耗；
    - 这种方式非常适合在资源受限或不想暴涨 Token 成本的情况下保留对话上下文。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


LOG_FILE = Path("chat.txt")


def log_messages(label: str, messages: List[Dict[str, str]]) -> None:
    """
    记录发送给大模型的完整消息：
        - label 用于标记是哪一种请求（聊天、摘要等），方便区分；
        - messages 是传给模型的消息列表，会包含 role 与 content；
        - 既打印到控制台，方便实时查看，也写入 chat.txt 持久化保存。
    """
    timestamp = datetime.now().isoformat(timespec="seconds")
    lines = [f"[{timestamp}] {label}"]
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    lines.append("")  # 末尾留空行，便于阅读
    entry = "\n".join(lines)
    try:
        with LOG_FILE.open("a", encoding="utf-8") as file_handle:
            file_handle.write(entry + "\n")
    except OSError as err:
        print(f"[LOG WARNING] 无法写入 {LOG_FILE}: {err}")


class ConversationSummaryMemory:
    """
    简易的 Conversation Summary Memory 实现：
        - 通过大模型不断对对话内容做摘要，使记忆保持精简；
        - 每次更新都会记录调用大模型时的完整 Prompt，方便排查问题；
        - 外部可以随时读取最新摘要，用于构造下一轮提示词。
    """

    def __init__(self, llm: ChatOpenAI) -> None:
        self._llm = llm
        self._summary: str = ""

    def load(self) -> str:
        """返回当前对话摘要文本（可能为空字符串）。"""
        return self._summary.strip()

    def update(self, user_message: str, assistant_reply: str) -> str:
        """
        接收最新一轮问答并刷新摘要。
        返回值为更新后的摘要，供调用方按需使用。
        """
        summary_prompt = (
            "你是一名对话摘要助手，需要维护一份精炼的中文摘要，供后续对话使用。\n"
            "当前的对话摘要如下：\n"
            f"{self._summary or '当前没有摘要。'}\n\n"
            "最新的用户消息：\n"
            f"{user_message}\n\n"
            "机器人刚刚的回复：\n"
            f"{assistant_reply}\n\n"
            "请给出更新后的中文摘要，保持简洁但不要遗漏关键需求与信息。"
        )
        summary_messages = [
            {"role": "system", "content": "你专注于编写简洁、信息完整的中文对话摘要。"},
            {"role": "user", "content": summary_prompt},
        ]
        log_messages("摘要更新", summary_messages)
        response = self._llm.invoke(summary_messages)
        self._summary = response.content if hasattr(response, "content") else str(response)
        return self.load()

    def clear(self) -> None:
        """清空摘要内容，可在需要时重置记忆。"""
        self._summary = ""


@dataclass
class DeepSeekChatbot:
    """
    DeepSeek 聊天机器人核心类，负责处理整套对话流程。
    字段说明：
        - api_key / api_base：调用 DeepSeek 必需的认证信息与接口地址；
        - model_name：要使用的对话模型名称，默认 deepseek-chat；
        - temperature：回答的发散程度，数值越低越保守；
        - system_prompt：对模型的整体行为设定，这里要求保持友好、简洁；
        - summary_memory：Conversation Summary Memory，用大模型维护精简摘要；
        - buffer_turns：额外保留多少轮最近问答，搭配摘要组成 Conversation Summary Memory；
        - recent_turns：实际缓存的最近问答列表，会在每轮对话后自动更新。
    通过把这些信息组合起来，我们就能在保持上下文的同时控制 Token 用量。
    """

    api_key: str
    api_base: str
    model_name: str = "deepseek-chat"
    temperature: float = 0.3
    system_prompt: str = (
        "你是一位温暖、耐心的中文助理，会主动结合上下文，为用户提供条理清晰、体贴入微的回复。"
        "请在回答时体现共情、鼓励与肯定，必要时给出明确的下一步行动建议。"
    )
    buffer_turns: int = 2  # 至多保留多少轮完整问答
    recent_turns: List[Dict[str, str]] = field(default_factory=list, init=False)
    _summary_memory: ConversationSummaryMemory = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # 创建聊天模型实例：负责生成回答的主要模型。
        # 使用者可以调整 temperature 或模型名称，以获得更符合需求的回复风格。
        self._chat_model = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            openai_api_key=self.api_key,
            openai_api_base=self.api_base,
        )
        # 创建摘要模型实例：专门用于压缩对话内容。
        # 这里依旧选择同一个 DeepSeek 模型，但降低 temperature，让摘要输出更稳定、更客观。
        self._summary_model = ChatOpenAI(
            model=self.model_name,
            temperature=0.2,
            openai_api_key=self.api_key,
            openai_api_base=self.api_base,
        )
        # 构建 Conversation Summary Memory，方便在下一轮对话时引用既往要点。
        self._summary_memory = ConversationSummaryMemory(self._summary_model)

    def chat(self, user_message: str) -> str:
        """
        处理单轮对话：
            1. 根据当前摘要、若干轮最近问答及本次输入构造请求消息；
            2. 调用 DeepSeek 获取回复；
            3. 用最新问答更新摘要和最近问答缓存，供下一次使用；
            4. 把回复返回给调用方。
        """
        # 拼装要发送给模型的消息列表，保证模型可以理解到目前的上下文。
        messages = self._build_messages(user_message)
        # 记录当前发送给大模型的完整 Prompt，既打印又写入 chat.txt。
        log_messages("对话请求", messages)
        # 通过 LangChain 的 ChatOpenAI 接口调用 DeepSeek，得到模型回复。
        response = self._chat_model.invoke(messages)
        assistant_reply = response.content if hasattr(response, "content") else str(response)
        # 根据最新问答刷新摘要，这一步能有效控制历史消息的长度。
        self._refresh_summary_memory(user_message, assistant_reply)
        # 更新最近问答缓存，确保最关键的上下文仍然保留。
        self._update_recent_turns(user_message, assistant_reply)
        return assistant_reply

    def _build_messages(self, user_message: str) -> List[Dict[str, str]]:
        """
        构建发送给模型的消息结构：
            - 第一条为系统提示，规定机器人整体行为和语气；
            - 若已有摘要，则追加一条系统消息，让模型知晓历史背景；
            - 若记录了最近几轮问答，则一并附加，帮助保持连续对话；
            - 最后将当前用户输入放入消息列表尾部。
        """
        # 系统提示是所有消息的基础，引导模型保持一致的风格。
        messages: List[Dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
        summary_text = self._summary_memory.load()
        if summary_text:
            # 将摘要说明注入到系统消息中，让模型理解整体进展。
            summary_prompt = (
                "截至目前的暖心对话摘要如下：\n"
                f"{summary_text}\n\n"
                "请在回复时参考该摘要，温柔地回应用户，并确保内容与既往对话保持一致。"
            )
            messages.append({"role": "system", "content": summary_prompt})
        if self.recent_turns:
            # 最近的若干轮问答能提供最新上下文，尽量保留以提高回答准确性。
            messages.extend(self.recent_turns)
        # 把当前问题放在最后，模型会据此生成新的回答。
        messages.append({"role": "user", "content": user_message})
        return messages

    def _refresh_summary_memory(self, user_message: str, assistant_reply: str) -> None:
        """
        利用 Conversation Summary Memory 刷新摘要，使记忆保持精简。
        """
        self._summary_memory.update(user_message, assistant_reply)

    def _update_recent_turns(self, user_message: str, assistant_reply: str) -> None:
        """
        维护最近若干轮问答：
            - 把最新的用户问题和机器人答案追加进缓存；
            - 只保留 buffer_turns 指定的轮数，以控制 Token 开销。
        """
        self.recent_turns.extend(
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_reply},
            ]
        )
        max_messages = max(self.buffer_turns, 0) * 2
        if max_messages and len(self.recent_turns) > max_messages:
            self.recent_turns = self.recent_turns[-max_messages:]


def ensure_api_credentials() -> DeepSeekChatbot:
    """
    读取并校验 DeepSeek 相关配置：
        - load_dotenv() 会自动读取 `.env` 文件，把其中的键值注入到环境变量；
        - 如果找不到 DEEPSEEK_API_KEY，则直接抛出错误，提醒用户配置；
        - 通过环境变量获取 API 网关地址，默认使用官方地址。
    """
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY is not set. Please export it before running the chatbot.")

    api_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
    return DeepSeekChatbot(api_key=api_key, api_base=api_base)


def main() -> None:
    """
    程序入口函数，职责如下：
        - 创建 DeepSeekChatbot 对象；若缺少密钥则给出明确提示；
        - 打印欢迎语，告诉用户可以输入 exit/quit 退出；
        - 循环读取用户输入，发送给聊天机器人并打印回复；
        - 捕获潜在异常（例如网络问题），避免程序直接崩溃。
    """
    try:
        bot = ensure_api_credentials()
    except ValueError as exc:
        print(exc)
        sys.exit(1)

    # 告知用户如何结束对话，会一直运行直到输入 exit/quit 或触发 EOF。
    print("DeepSeek Chatbot (type 'exit' to quit)")
    while True:
        try:
            # input() 会等待用户输入，strip() 用来清除多余的首尾空格，确保处理整洁的字符串。
            user_input = input("You: ").strip()
        except EOFError:
            # 捕获 Ctrl+D（Unix）或类似操作触发的 EOF，优雅退出。
            print("\nGoodbye!")
            break

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not user_input:
            # 空输入时给出提示，避免把空字符串发给模型浪费请求次数。
            print("Assistant: Please enter a question or type 'exit' to quit.")
            continue

        try:
            # 调用聊天机器人处理用户输入，这里可能会触发网络异常。
            reply = bot.chat(user_input)
        except Exception as err:
            # 捕获异常并反馈给用户，方便快速定位问题（例如 API Key 有误或网络波动）。
            print(f"Assistant: An error occurred while contacting DeepSeek: {err}")
            continue

        # 正常返回时，把模型回答展示到命令行。
        print(f"Assistant: {reply}")


if __name__ == "__main__":
    main()
