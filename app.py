# app.py
import chainlit as cl
from chainlit.input_widget import Select, Switch, Slider

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer, Memory
from llama_index.llms.openai import OpenAI

import logging

from llama_index.vector_stores.supabase import SupabaseVectorStore

logger = logging.getLogger(__name__)

llm = OpenAI(model="gpt-4o-mini", temperature=0)
data = SimpleDirectoryReader(input_dir="./data/paul_graham/").load_data()
vector_store = SupabaseVectorStore(
    postgres_connection_string=DB_CONNECTION,
    collection_name='hrbrunei_files'
)
index = VectorStoreIndex.from_documents(data)

@cl.on_chat_start
async def start():
    """Handler for chat start events. Sets session variables."""
    system_prompt = SYSTEM_PROMPTS[chat_profile]
    memory = ChatMemoryBuffer.from_defaults()
    memory.put(
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=system_prompt
        )
    )
    cl.user_session.set(
        "agent",
        SimpleChatEngine.from_defaults(
            llm=openai_llm,
        )
    )
    memory = Memory.from_defaults()
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