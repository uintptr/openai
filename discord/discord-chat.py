#!/usr/bin/env python3

import sys
import os
import io
import json
import time
import copy
from typing import List, Optional
from dataclasses import dataclass
from typing import Callable, Awaitable
from datetime import datetime
import smtplib
import asyncio
import openai
import tempfile

import aiohttp
from bs4 import BeautifulSoup
import discord
import PyPDF2
from discord.message import Message

GPT_3_5_COST = 0.002
GPT_CODE_MODEL = "code-davinci-002"
GPT_CHAT_MODEL = "gpt-3.5-turbo-16k"

DISCORD_MAX_LENGTH = 2000


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
        self.channels = {}

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

    async def get(self, channel_id: int, max: int = 10) -> List[ChatHistory]:

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

    def get(self, component: str, key: str, default: Optional[str] = None) -> str:

        if (component in self.config):
            if (key in self.config[component]):
                return self.config[component][key]
        else:
            raise ConfigNotFound(f'component "{key}" is not defined')

        if (default is not None):
            return default

        raise ConfigNotFound(f"{key} is not defined")

    def get_int(self, component: str, key: str, default: Optional[int] = None) -> int:

        default_str = None

        if (default is not None):
            default_str = str(default)
        else:
            default_str = None

        return int(self.get(component, key, default_str))

    def get_float(self, component: str, key: str, default: Optional[float] = None) -> float:

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
                             "gpt-4"]

        self.url = config.get("openai", "url")
        self.key = config.get("openai", "key")
        self.model = config.get("openai", "model")
        self.system = config.get("openai", "system")
        self.temperature = config.get_float("openai", "temperature", 0.3)
        self.max_prompt = config.get_int("openai", "max_prompt", 12)

        self.stats = ChatStats()
        self.stats_lock = asyncio.Lock()
        self.config = config

        openai.api_key = self.key

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

    def get_known_models(self) -> List[str]:
        return self.known_models

    def to_dict(self) -> dict:

        config = {}
        config["model"] = self.model
        config["system"] = self.system
        config["temperature"] = self.temperature
        config["max_prompt"] = self.max_prompt
        return config

    async def image(self, prompt: str, num: int = 2, size: str = "1024x1024") -> List[bytes]:

        images = []

        comp = await openai.Image.acreate(prompt=prompt, n=num, size=size)

        resp = comp.to_dict()  # type: ignore

        if ("data" in resp):
            for url in resp["data"]:
                images.append(await download_url(url.url))

        return images

    async def chat(self, history: List[ChatHistory], user: Optional[str] = None) -> ChatResponse:

        messages = [
            {"role": "system", "content": self.system}
        ]

        history = history[-self.max_prompt:]

        for h in history:
            message = {"role": h.role, "content": h.content}
            messages.append(message)

        create_ts = time.time()

        completion = await openai.ChatCompletion.acreate(model=self.model,
                                                         temperature=self.temperature,
                                                         messages=messages)

        response_ts = time.time()

        data = completion.to_dict()  # type: ignore

        id = ""
        message = ""

        if ("usage" in data):
            usage = data["usage"]
            prompts = usage["prompt_tokens"]
            completions = usage["completion_tokens"]

            async with self.stats_lock:
                self.stats.update(prompts, completions)

        if ("id" in data):
            id = data["id"]

        if ("choices" in data):
            message = data["choices"][0]["message"]["content"]
        elif ("error" in data):
            message = data["error"]["message"]

        return ChatResponse(create_ts, response_ts, id, message)

    def update_system(self, system: str):
        self.system = system

    async def get_stats(self) -> ChatStats:
        async with self.stats_lock:
            return copy.copy(self.stats)


class ChatDiscord(discord.Client):

    def __init__(self, ai: AIApi, config: Config, *args, **kwargs) -> None:

        self.ai = ai
        self.bot_token = config.get("discord", "token")
        self.start_time = time.time()
        self.max_age = config.get_int("discord", "max_age")
        self.file_size_max = config.get_int("discord", "file_size_max")
        self.gmail_email = config.get("gmail", "email")
        self.gmail_password = config.get("gmail", "password")

        self.history = History()

        self.handlers: List[CommandHandler] = [
            CommandHandler("help", self.cmd_help, "This command"),
            CommandHandler("ping", self.cmd_ping, "Display Pong"),
            CommandHandler("clear", self.cmd_clear, "Clear Message History"),
            CommandHandler("reset", self.cmd_clear, "Clear Message History"),
            CommandHandler("config", self.cmd_config, "Display Server Config"),
            CommandHandler("uptime", self.cmd_uptime, "Display Server uptime"),
            CommandHandler("email", self.cmd_email, "Email Logs"),
            CommandHandler("stats", self.cmd_stats, "Stats for nerds"),
            CommandHandler("history", self.cmd_history, "Display history"),
            CommandHandler("model", self.cmd_model, "Get and set model"),
            CommandHandler("models", self.cmd_models, "List models"),
            CommandHandler("code", self.cmd_code, "Switch to code model"),
            CommandHandler("chat", self.cmd_chat, "Switch to chat model"),
            CommandHandler("image", self.cmd_image, "Generate an image"),
        ]

        super().__init__(*args, **kwargs)

    def to_dict(self) -> dict:

        config = {}
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
    # CODE
    ##################

    async def cmd_code(self, msg: Message) -> str:
        self.ai.set_model(GPT_CODE_MODEL)
        return f"Changed model to `{GPT_CODE_MODEL}`"

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
    # EMAIL
    ##################
    async def cmd_email(self, msg: Message) -> str:

        response = ""
        content = msg.content.split(' ')

        if (2 == len(content)):

            recipient = content[1]

            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(self.gmail_email, self.gmail_password)

                message = "message"
                subject = "subject"
                body = "Subject: {}\n\n{}".format(subject, message)

                server.sendmail("your_email@gmail.com", recipient, body)
        else:
            response = "Usage:\n/email dest@email.com"

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
        return f"whatup {msg.author.mention} :stuck_out_tongue_winking_eye:"

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
    def __load_pdf(self, content: bytes) -> str:

        pdf_content = ""

        with io.BytesIO(content) as f:
            reader = PyPDF2.PdfReader(f)

            for i in range(len(reader.pages)):

                page = reader.pages[i]
                pdf_content += page.extract_text()

        return pdf_content

    async def __load_attachments(self, msg) -> str:

        data = ""

        for a in msg.attachments:

            ext = os.path.splitext(a.filename)[1].lower()

            content = await a.read()

            data += f"consider the following document as {a.filename}\n"

            if (ext.endswith(".txt")):
                data += content.encode("utf-8")
            elif (ext.endswith(".pdf")):
                data += self.__load_pdf(content)

        return data

    def __parse_html(self, html_content) -> str:

        soup = BeautifulSoup(html_content, 'html.parser')
        # You can perform further parsing or extraction here
        # For example, you can find elements using soup.find() or soup.find_all()
        return soup.get_text()

    async def __download_page(self, url) -> str:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                html_content = await response.text()
                return html_content

    async def __load_url(self, msg) -> str:

        url = msg.content

        html_content = await self.__download_page(url)

        html_text = f"consider the follow data as {url}\n\n"

        html_text += self.__parse_html(html_content)

        return html_text

    async def __command_handler(self, msg: Message) -> str:

        cmd = msg.content.split()[0][1:]

        for h in self.handlers:

            if (cmd == h.name):
                return await h.handler(msg)

        return "command not found"

    async def __chat_handler(self, msg: Message) -> str:

        response = ""

        # load attachments (if any)
        if (len(msg.attachments) > 0):
            content = await self.__load_attachments(msg)

        elif (msg.content.startswith("http://") or
                msg.content.startswith("https://")):
            content = await self.__load_url(msg)
        else:
            content = msg.content

        if (len(content) > 0):

            await self.history.add(msg.channel.id, "user", content)

            history = await self.history.get(msg.channel.id,
                                             self.ai.max_prompt)

            if (len(history) > 0):
                chat = await self.ai.chat(history, msg.author.name)

                await self.history.add(msg.channel.id,
                                       "assistant",
                                       chat.message)

                response = chat.message

        return response

    async def __msg_handler(self, msg: Message) -> str:

        if (msg.content.startswith("/")):
            response = await self.__command_handler(msg)
        else:
            response = await self.__chat_handler(msg)

        return response

    ############################################################################
    # DISCORD CALLBACKS
    ############################################################################

    async def on_ready(self):
        print('Logged on as', self.user)

    async def on_message(self, msg):

        # ignore self
        if msg.author == self.user:
            return

        try:

            async with msg.channel.typing():

                response = await self.__msg_handler(msg)
                rlen = len(response)

                if (rlen > 0):
                    for i in range(0, rlen, DISCORD_MAX_LENGTH):
                        part = response[i:i+DISCORD_MAX_LENGTH]
                        await msg.channel.send(part)

        except Exception as e:
            await msg.channel.send(f"Exception: {e}")


def main() -> int:

    status = 1

    try:

        config = Config()

        ai = AIApi(config)

        # init the intent because starting the discord client
        intents = discord.Intents.default()
        intents.message_content = True

        client = ChatDiscord(ai, config, intents=intents)

        client.run(config.get("discord", "token"))

        status = 0
    except KeyboardInterrupt:
        status = 0
    except ConfigNotFound as e:
        print(e)

    return status


if __name__ == '__main__':

    status = main()

    if status != 0:
        sys.exit(status)
