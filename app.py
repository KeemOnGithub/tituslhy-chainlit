# app.py
import json
from typing import Optional

import chainlit as cl
import vecs
from chainlit.input_widget import Select, Switch, Slider

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.chat_engine.types import ChatMode
from llama_index.core.memory import ChatMemoryBuffer, Memory
from llama_index.llms.openai import OpenAI

import logging

from llama_index.vector_stores.supabase import SupabaseVectorStore

logger = logging.getLogger(__name__)

with open("system_prompts.json", encoding='utf-8') as json_file:
    SYSTEM_PROMPTS = json.load(json_file)

llm = OpenAI(model="gpt-4o-mini", temperature=0)

DB_CONNECTION = "postgresql://postgres.thtacujdwcbdxuzqyidl:LQhaireel107@aws-0-ap-southeast-1.pooler.supabase.com:5432/postgres"

vector_store = SupabaseVectorStore(
    postgres_connection_string=DB_CONNECTION,
    collection_name='pitchmaster_files'
)

index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
# Add this debug code to check if documents exist
print(f"Number of documents in index: {len(index.docstore.docs)}")
chat_engine = index.as_chat_engine(chat_mode=ChatMode.BEST, llm=llm, verbose=True)

@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
    """Password auth handler for login"""

    if (username, password) == ("admin", "admin"):
        return cl.User(identifier="admin", metadata={"role": "ADMIN"})
    else:
        return None

@cl.set_chat_profiles
async def chat_profile():
    """Chat profile setter."""

    return [
        cl.ChatProfile(
            name="GELIGA HR",
            markdown_description="This LLM is your personal HR assistant.",
            icon="public/assistant.png"
        ),
        cl.ChatProfile(
            name="GELIGA PitchMaster",
            markdown_description="This LLM is a product pitching assistant.",
            icon="public/cowboy.png"
        )
    ]

@cl.on_chat_start
async def start():
    """Handler for chat start events. Sets session variables."""
    cl.user_session.set(
        "agent",
        chat_engine
    )
    memory = ChatMemoryBuffer.from_defaults()
    cl.user_session.set("memory", memory)
    chat_profile = cl.user_session.get("chat_profile")
    user = cl.user_session.get("user")
    logger.info(f"{user.identifier} has started the conversation")

    system_prompt = SYSTEM_PROMPTS[chat_profile]
    memory.put(
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=system_prompt
        )
    )
    cl.user_session.set("memory", memory)
    settings = await cl.ChatSettings(
        [
            Select(
                id="LLM",
                label="OpenAI model to use",
                values=["gpt-4o-mini", "gpt-4o"],
                initial_index=0,
            )
        ]
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """On message handler to handle message received events"""

    agent = cl.user_session.get("agent")
    memory = cl.user_session.get("memory")
    chat_profile = cl.user_session.get("chat_profile")
    chat_history = memory.get()
    msg = cl.Message("")

    # Get the LLM to stream replies back!
    response = agent.stream_chat(message.content, chat_history=chat_history)
    for token in response.response_gen:
        await msg.stream_token(token)

    # Update chat history
    memory.put(
        ChatMessage(
            role=MessageRole.USER,
            content=message.content
        )
    )
    memory.put(
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content=str(response)
        )
    )
    cl.user_session.set("memory", memory)

    await msg.send()