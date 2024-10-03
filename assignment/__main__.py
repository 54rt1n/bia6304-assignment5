# assignment/__main__.py

import click
from dataclasses import asdict
import json

from .chat import ChatManager
from .config import ChatConfig, ENV_CONFIG
from .dqm import DocumentQueryModel
from .llm import LLMProvider

@click.group()
@click.option('--embedding-model', default=ENV_CONFIG['embedding_model'], help='Embedding model to use')
@click.option('--db-path', default=ENV_CONFIG['db_path'], help='path for db')
@click.pass_context
def cli(ctx, embedding_model, db_path):
    config = ChatConfig(
        embedding_model=embedding_model,
        db_path=db_path,
    )
    ctx.obj = DocumentQueryModel.from_config(config)

@cli.command()
@click.option('--embedding-model', default=ENV_CONFIG['embedding_model'], help='Embedding model to use')
@click.option('--db-path', default=ENV_CONFIG['db_path'], help='path for db')
@click.option('--llm-provider', default=ENV_CONFIG['llm_provider'], help='LLM provider to use: ai_studio, openai, groq or cohere')
@click.option('--model-url', default=ENV_CONFIG['model_url'], help='URL for the OpenAI-compatible API')
@click.option('--api-key', default=ENV_CONFIG['api_key'], help='API key for the LLM service')
@click.option('--user-id', default=ENV_CONFIG['user_id'], help='User ID for the conversation')
@click.option('--system-message', default=ENV_CONFIG['system_message'], help='System message for the chat')
@click.option('--max-tokens', default=ENV_CONFIG['max_tokens'], help='Maximum number of tokens for LLM response')
@click.option('--temperature', default=ENV_CONFIG['temperature'], help='Temperature for LLM response')
@click.pass_obj
def chat(dqm: DocumentQueryModel, embedding_model, db_path, llm_provider, model_url, api_key, user_id, system_message, max_tokens, temperature):
    """Start a new chat session"""

    config = ChatConfig(
        embedding_model=embedding_model,
        db_path=db_path,
        llm_provider=llm_provider,
        model_url=model_url,
        api_key=api_key,
        system_message=system_message,
        user_id=user_id,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    llm = LLMProvider.from_config(config)

    cm = ChatManager(llm=llm, dqm=dqm, config=config, clear_output=lambda: click.clear())
    cm.chat_loop()

@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.pass_obj
def load_area_data(dqm: DocumentQueryModel, file_path):
    dqm.load_jsonl(file_path)
    dqm.save()
    click.echo(f"Data loaded from {file_path}")

if __name__ == '__main__':
    cli()
