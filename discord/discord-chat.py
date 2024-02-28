#!/usr/bin/env python3

import sys
import os
import json
import time
import copy
from typing import Any
from dataclasses import dataclass
from typing import Callable, Awaitable
from datetime import datetime
from enum import Enum
import asyncio
import tempfile
import traceback

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat import ChatCompletionUserMessageParam
from openai.types.chat import ChatCompletionSystemMessageParam
from openai.types.chat import ChatCompletionAssistantMessageParam

import aiohttp
import discord
from discord.message import Message

GPT_3_5_COST = 0.002
GPT_CODE_MODEL = "code-davinci-002"
GPT_CHAT_MODEL = "gpt-3.5-turbo-16k"

DISCORD_MAX_LENGTH = 2000


class ChatAttachmentType(Enum):
    NONE = 0
    TEXT_FILE = 1
    AUDIO_FILE = 2


@dataclass
class ChatResponse:
    create_ts: float
    response_ts: float
    id: str
    message: str

    def __str__(self):
        return f"id=${self.id} message={self.message}"


@dataclass
class CommandHandler:
    name: str
    handler: Callable[[Message], Awaitable[str]]
    doc: str = "Not documented"
    args: bool = False


@dataclass
class ChatHistory:
    role: str
    content: str
    ts: float


@dataclass
class ChatStats:
    queries: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def update(self, prompts: int, completions: int) -> None:
        self.queries += 1
        self.prompt_tokens += prompts
        self.completion_tokens += completions


def printkv(k: str, v: object) -> None:

    k = f"{k}:"
    print(f"    {k:<25}{v}")


class ConfigNotFound(Exception):
    pass


class ChatModelNotFound(Exception):
    pass


class HttpStatusError(Exception):
    pass


async def download_url(url: str) -> bytes:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status == 200:
                return await resp.read()

            raise HttpStatusError(f"status={resp.status}")


class History:

    def __init__(self) -> None:

        self.history_lock = asyncio.Lock()
        self.channels: dict[int, list[ChatHistory]] = {}

    async def clear(self, channel_id: int) -> None:

        async with self.history_lock:
            if (channel_id in self.channels):
                self.channels[channel_id] = []

    async def add(self, channel_id: int, role: str, content: str) -> None:

        ts = time.time()

        async with self.history_lock:
            if (channel_id not in self.channels):
                self.channels[channel_id] = []

            entry = ChatHistory(role, content, ts)
            self.channels[channel_id].append(entry)

    async def get(self, channel_id: int, max: int = 10) -> list[ChatHistory]:

        async with self.history_lock:
            if (channel_id in self.channels):

                truncated = self.channels[channel_id][-max:]

                return copy.copy(truncated)

        return []


class Config:

    def __init__(self) -> None:
        script_root = os.path.abspath(os.path.dirname(sys.argv[0]))

        config_file = os.path.join(script_root, "config.json")

        if (False == os.path.exists(config_file)):
            config_file = os.path.join(script_root, "..", "config.json")

        with open(config_file, "r") as f:
            self.config = json.load(f)

        self.config_file = config_file

    def __sync(self) -> None:

        with open(self.config_file, "w+") as f:
            f.write(json.dumps(self.config, indent=4))

    def get(self, component: str, key: str, default: str | None = None) -> str:

        if (component in self.config):
            if (key in self.config[component]):
                return self.config[component][key]
        else:
            raise ConfigNotFound(f'component "{key}" is not defined')

        if (default is not None):
            return default

        raise ConfigNotFound(f"{key} is not defined")

    def get_int(self, component: str, key: str, default: int | None = None) -> int:

        default_str = None

        if (default is not None):
            default_str = str(default)
        else:
            default_str = None

        return int(self.get(component, key, default_str))

    def get_float(self, component: str, key: str, default: float | None = None) -> float:

        default_str = None

        if (default is not None):
            default_str = str(default)
        else:
            default_str = None

        return float(self.get(component, key, default_str))

    def set(self, component: str, key: str, value: str) -> None:

        if (component in self.config):
            self.config[component][key] = value
        else:
            raise ConfigNotFound(f'component "{key}" is not defined')

        self.__sync()


class AIApi:
    def __init__(self, config: Config) -> None:

        self.known_models = ["gpt-3.5-turbo",
                             "gpt-3.5-turbo-16k",
                             "code-davinci-002",
                             "gpt-4-1106-preview"]

        self.url = config.get("openai", "url")
        self.key = config.get("openai", "key")
        self.model = config.get("openai", "model")
        self.system = config.get("openai", "system")
        self.temperature = config.get_float("openai", "temperature", 0.3)
        self.max_prompt = config.get_int("openai", "max_prompt", 12)

        self.stats = ChatStats()
        self.stats_lock = asyncio.Lock()
        self.config = config

        self.client = AsyncOpenAI(api_key=self.key)

    def __str__(self) -> str:
        return f"model={self.model} system={self.system}"

    def get_model(self) -> str:
        return self.model

    def set_model(self, model: str) -> None:

        if (model in self.known_models):
            self.model = model
            self.config.set("openai", "model", model)
        else:
            raise ChatModelNotFound(f'unknown model "{model}"')

    def get_known_models(self) -> list[str]:
        return self.known_models

    def to_dict(self) -> dict[str, Any]:

        config: dict[str, Any] = {}
        config["model"] = self.model
        config["system"] = self.system
        config["temperature"] = self.temperature
        config["max_prompt"] = self.max_prompt
        return config

    async def image(self, prompt: str, num: int = 1) -> list[bytes]:

        images: list[bytes] = []

        res = await self.client.images.generate(model="dall-e-3",
                                                prompt=prompt,
                                                size="1024x1024",
                                                quality="standard",
                                                n=num)

        for data in res.data:
            if (data.url is not None):
                images.append(await download_url(data.url))

        return images

    async def speech_to_text(self, file_path: str) -> str:

        with open(file_path, "rb") as f:

            res = await self.client.audio.transcriptions.create(model="whisper-1",
                                                                file=f)
            return res.text

    async def text_to_speech(self, text: str, out_file: str) -> None:

        res = await self.client.audio.speech.create(model="tts-1",
                                                    voice="onyx",
                                                    response_format="aac",
                                                    input=text)

        await res.astream_to_file(out_file)

    async def chat(self, history: list[ChatHistory], user: str | None = None) -> ChatResponse:

        messages: list[ChatCompletionMessageParam] = []

        s = ChatCompletionSystemMessageParam(
            content=self.system, role="system")

        messages.append(s)

        history = history[-self.max_prompt:]

        for h in history:

            if ("user" == h.role):
                m = ChatCompletionUserMessageParam(
                    content=h.content, role="user")
            else:
                m = ChatCompletionAssistantMessageParam(
                    content=h.content, role="assistant")

            messages.append(m)

        create_ts = time.time()

        comp = await self.client.chat.completions.create(model=self.model,
                                                         temperature=self.temperature,
                                                         messages=messages)

        response_ts = time.time()

        id = ""
        message = ""

        if (comp.usage is not None):
            prompts = comp.usage.prompt_tokens
            completions = comp.usage.completion_tokens

            async with self.stats_lock:
                self.stats.update(prompts, completions)

        id = comp.id

        if (comp.choices[0].message.content is not None):
            message = comp.choices[0].message.content
        else:
            message = "empty response from completions API"

        return ChatResponse(create_ts, response_ts, id, message)

    def update_system(self, system: str) -> None:
        self.system = system

    async def get_stats(self) -> ChatStats:
        async with self.stats_lock:
            return copy.copy(self.stats)


class ChatDiscord(discord.Client):

    def __init__(self, ai: AIApi, config: Config, *args: Any, **kwargs: Any) -> None:

        self.ai = ai
        self.bot_token = config.get("discord", "token")
        self.start_time = time.time()
        self.max_age = config.get_int("discord", "max_age")
        self.file_size_max = config.get_int("discord", "file_size_max")

        self.history = History()

        self.handlers: list[CommandHandler] = [
            CommandHandler("help", self.cmd_help, "This command"),
            CommandHandler("ping", self.cmd_ping, "Display Pong"),
            CommandHandler("clear", self.cmd_clear, "Clear Message History"),
            CommandHandler("reset", self.cmd_clear, "Clear Message History"),
            CommandHandler("config", self.cmd_config, "Display Server Config"),
            CommandHandler("uptime", self.cmd_uptime, "Display Server uptime"),
            CommandHandler("stats", self.cmd_stats, "Stats for nerds"),
            CommandHandler("history", self.cmd_history, "Display history"),
            CommandHandler("model", self.cmd_model, "Get and set model"),
            CommandHandler("models", self.cmd_models, "List models"),
            CommandHandler("chat", self.cmd_chat, "Switch to chat model"),
            CommandHandler("image", self.cmd_image, "Generate an image"),
        ]

        super().__init__(*args, **kwargs)

    def to_dict(self) -> dict[str, Any]:

        config: dict[str, Any] = {}
        config["max_age"] = self.max_age
        config["file_size_max"] = self.file_size_max

        return config

    ############################################################################
    # COMMANDS
    ############################################################################

    ##################
    # IMAGE
    ##################
    async def cmd_image(self, msg: Message) -> str:

        i = 0

        prompt = msg.content

        for img in await self.ai.image(prompt):

            with tempfile.TemporaryDirectory(prefix="openai_image_") as td:

                tmp_file = os.path.join(td, "image.png")

                with open(tmp_file, "wb+") as f:
                    f.write(img)
                    f.flush()
                    f.seek(0)

                    file = discord.File(f)
                    await msg.channel.send(file=file)

            i += 1

        return ""

    ##################
    # CHAT
    ##################
    async def cmd_chat(self, msg: Message) -> str:
        self.ai.set_model(GPT_CHAT_MODEL)
        return f"Changed model to `{GPT_CHAT_MODEL}`"

    ##################
    # MODEL
    ##################
    async def cmd_model(self, msg: Message) -> str:

        model = msg.content[6:].strip(" ")

        if (len(model) > 0):
            self.ai.set_model(model)
            response = f"Model changed to `{model}`"
        else:
            model = self.ai.get_model()
            response = f"Current model is `{model}`"

        return response

    ##################
    # MODELS
    ##################
    async def cmd_models(self, msg: Message) -> str:

        response = "Models:\n"

        for m in self.ai.get_known_models():
            response += f"* `{m}`\n"

        return response

    ##################
    # HISTORY
    ##################
    async def cmd_history(self, msg: Message) -> str:

        history = await self.history.get(msg.channel.id, self.ai.max_prompt)

        if (history == []):
            return "Nothing to see..."

        reply = ""
        idx = 1

        for h in history:

            reply += "`───────────────────────────────────`\n"
            reply += f"`#{idx} {h.role}`\n"
            reply += h.content + "\n"

            idx += 1

        return reply

    ##################
    # STATS
    ##################
    async def cmd_stats(self, msg: Message) -> str:

        s = await self.ai.get_stats()

        total = s.prompt_tokens + s.completion_tokens

        price = (total * GPT_3_5_COST) / 1000
        price_str = f"{price:.5f}¢"

        response = "```\n"
        response += "╭───────────────────────────────╮\n"
        response += "│ Type              │  Count    │\n"
        response += "│───────────────────────────────│\n"
        response += f"│ Queries           │ {s.queries: 9} │\n"
        response += f"│ Prompt Tokens     │ {s.prompt_tokens: 9} │\n"
        response += f"│ Completion Tokens │ {s.completion_tokens: 9} │\n"
        response += f"│ Total Tokens      │ {total: 9} │\n"
        response += f"│ Cost              │ {price_str:>9} │\n"
        response += "╰───────────────────────────────╯\n"
        response += "```\n"

        return response

    ##################
    # UPTIME
    ##################
    async def cmd_uptime(self, msg: Message) -> str:
        uptime = datetime.now() - datetime.fromtimestamp(self.start_time)

        days = uptime.days
        hours, remainder = divmod(uptime.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if days == 0:
            uptime_formatted = '{:02d}:{:02d}:{:02d}'.format(
                hours, minutes, seconds)
        else:
            uptime_formatted = '{} day{}, {:02d}:{:02d}:{:02d}'.format(
                days, '' if days == 1 else 's', hours, minutes, seconds)
        return uptime_formatted

    ##################
    # CLEAR
    ##################

    async def cmd_clear(self, msg: Message) -> str:

        await self.history.clear(msg.channel.id)

        if isinstance(msg.channel, discord.TextChannel):
            await msg.add_reaction('😏')
            await msg.channel.purge(limit=msg.id + 1)
        else:
            pass  # not supported for DMs

        model = self.ai.get_model()
        return f"Hello World (`{model}`)"

    ##################
    # PING
    ##################
    async def cmd_ping(self, msg: Message) -> str:
        return f"whatup {msg.author.mention} ? :stuck_out_tongue_winking_eye:"

    ##################
    # HELP
    ##################
    async def cmd_help(self, msg: Message) -> str:

        message = "Available commands:\n"

        for h in self.handlers:
            message += f"    `/{h.name:<9}` {h.doc}\n"

        return message

    ##################
    # CONFIG
    ##################
    async def cmd_config(self, msg: Message) -> str:

        config = {}
        config["chat"] = self.ai.to_dict()
        config["discord"] = self.to_dict()

        config_json = json.dumps(config, indent=4)

        return f"```json\n{config_json}```"

    ############################################################################
    # PRIVATE
    ############################################################################

    async def __load_attachments(self, msg: Message) -> tuple[ChatAttachmentType, str]:

        data: str = ""
        atype = ChatAttachmentType.NONE

        for a in msg.attachments:

            ext = os.path.splitext(a.filename)[1].lower()

            if (ext.endswith(".txt")):
                content = await a.read()
                data += f"consider the following document as {a.filename}\n"
                data += content.decode("utf-8")
                atype = ChatAttachmentType.TEXT_FILE
            elif (ext.endswith(".ogg")):

                with tempfile.TemporaryDirectory(prefix="ogg_") as td:
                    input_ogg_file = os.path.join(td, "stt.ogg")

                    with open(input_ogg_file, "wb+") as f:
                        f.write(await a.read())

                    data = await self.ai.speech_to_text(input_ogg_file)

                    await msg.channel.send(f"input: {data}")
                    atype = ChatAttachmentType.AUDIO_FILE
            else:
                NotImplementedError(f"{ext} is not implemented")

        return atype, data

    async def __command_handler(self, msg: Message) -> str:

        cmd = msg.content.split()[0][1:]

        for h in self.handlers:

            if (cmd == h.name):
                return await h.handler(msg)

        return "command not found"

    async def __chat_handler(self, msg: Message) -> str:

        response = ""
        atype = ChatAttachmentType.NONE

        # load attachments (if any)
        if (len(msg.attachments) > 0):
            atype, content = await self.__load_attachments(msg)
        else:
            content = msg.content

        if (content == ""):
            return ""

        await self.history.add(msg.channel.id, "user", content)

        history = await self.history.get(msg.channel.id,
                                         self.ai.max_prompt)

        if (len(history) > 0):
            chat = await self.ai.chat(history, msg.author.name)

            await self.history.add(msg.channel.id, "assistant", chat.message)

            if (atype == ChatAttachmentType.AUDIO_FILE):

                with tempfile.TemporaryDirectory(prefix="tts_") as td:

                    audio_file = os.path.join(td, "tts.aac")

                    await self.ai.text_to_speech(chat.message, audio_file)

                    await msg.channel.send(file=discord.File(audio_file))

            response = chat.message

        return response

    async def __msg_handler(self, msg: Message) -> str:

        response = ""

        if (msg.content.startswith("/")):
            response = await self.__command_handler(msg)
        else:
            response = await self.__chat_handler(msg)

        return response

    ############################################################################
    # DISCORD CALLBACKS
    ############################################################################

    async def on_ready(self) -> None:
        print('Logged on as', self.user)

    async def on_message(self, msg: Message) -> None:

        # ignore self
        if msg.author == self.user:
            return

        async with self._last_chat_ts_lock:
            self._last_chat_ts = time.time()

        try:

            async with msg.channel.typing():

                try:
                    response = await self.__msg_handler(msg)
                except Exception:
                    exception_stack = traceback.format_exc()
                    response = ":crying_cat_face:\n"
                    response += f"```\n{exception_stack}\n```\n"

                rlen = len(response)

                if (rlen > 0):
                    for i in range(0, rlen, DISCORD_MAX_LENGTH):
                        part = response[i:i+DISCORD_MAX_LENGTH]
                        await msg.channel.send(part)

        except Exception as e:
            await msg.channel.send(f"Exception: {e}")


async def amain() -> int:

    status = 1

    try:

        config = Config()

        ai = AIApi(config)

        # init the intent because starting the discord client
        intents = discord.Intents.default()
        intents.message_content = True

        async with ChatDiscord(ai, config, intents=intents) as client:
            await client.start(config.get("discord", "token"))

        #
        # await asyncio.gather(
        #    asyncio.to_thread(
        # asyncio.sleep(1))

        status = 0

    except ConfigNotFound as e:
        print(e)

    return status


if __name__ == '__main__':

    try:
        status = asyncio.run(amain())
    except KeyboardInterrupt:
        status = 1

    if status != 0:
        sys.exit(status)
