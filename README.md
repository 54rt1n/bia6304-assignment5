# Assignment 5: Retrieval Augmented Generation

## Overview

This project implements a Retrieval Augmented Generation (RAG) system using a document query model and a large language model. It demonstrates proficiency in natural language processing, vector embeddings, and conversational AI. The system allows users to chat with an AI assistant that can retrieve relevant information from a document database to enhance its responses.

## Key Features

- Document indexing and querying using vector embeddings
- Integration with various Large Language Model (LLM) providers
- Retrieval Augmented Generation for enhanced AI responses
- Interactive chat interface with both Jupyter notebook and command-line support
- Customizable chat strategies and configurations

## Technology Stack

- Python 3.9+
- Poetry for dependency management
- Hugging Face Transformers for embeddings
- Google AI Studio, OpenAI (for OpenAI's API (not free), or compatible local models), or Groq for LLM integration
- Pandas for data management
- Click for command-line interface

## Project Structure

```
assignment5/
├── assets/
│   └── area_results.jsonl
├── assignment/
│   ├── __init__.py
│   ├── assignment5.py
│   ├── chat.py
│   ├── config.py
│   ├── dqm.py
│   ├── embedding.py
│   ├── llm.py
│   └── __main__.py
├── 01-load.ipynb
├── 02-run.ipynb
├── pyproject.toml
└── README.md
```

## Setup

1. Ensure you have Python 3.9 or higher installed.
2. Install Poetry if you haven't already:
   ```
   pip install poetry
   ```
3. Clone the repository:
   ```
   git clone https://github.com/yourusername/bia6304-assignment5.git
   cd bia6304-assignment5
   ```
4. Install dependencies:
   ```
   poetry install
   ```
5. Sign up for a Google AI Studio API key:
   - Visit https://ai.google.dev/aistudio
   - Sign in with your Google account or create one
   - Navigate to the API section and generate an API key
6. Set up your environment variables in a `.env` file:
   ```
   EMBEDDING_MODEL=Snowflake/snowflake-arctic-embed-l
   DB_PATH=data.pkl
   LLM_PROVIDER=ai_studio
   API_KEY=your_google_ai_studio_api_key_here
   ```
You can use whatever embedding model you find to be best.  Consult the [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) for a list of popular models.

## Running the Application

This project provides two methods for running the RAG system: Jupyter notebooks and a command-line interface (CLI). Both methods accomplish the same task of loading data, indexing documents, and running the chat interface.

### Jupyter Notebooks

The project includes two Jupyter notebooks:

1. `01-load.ipynb`:
   - Used for loading and indexing documents
   - Demonstrates how to use the DocumentQueryModel (DQM) to process and store document data

2. `02-run.ipynb`:
   - Initializes the chat interface
   - Allows you to interact with the RAG system in an interactive notebook environment

To use the notebooks:

1. Start Jupyter Lab or Jupyter Notebook.
2. Open and run `01-load.ipynb` to load and index your documents.
3. Open and run `02-run.ipynb` to start the chat interface.

### Command-Line Interface (CLI)

The project also provides a CLI for those who prefer working in the terminal or want to integrate the system into scripts.

To use the CLI:

1. To load area data:
   ```
   poetry run python -m assignment load-area-data <path_to_jsonl_file>
   ```

2. To start a chat session:
   ```
   poetry run python -m assignment chat
   ```

### Choosing Between Notebooks and CLI

Both methods (notebooks and CLI) achieve the same end result. Choose the method that best fits your workflow:

- Use the notebooks if you prefer a visual, interactive environment or want to experiment with the code step-by-step.
- Use the CLI if you're comfortable with command-line tools, need to automate the process, or want to integrate the system into larger scripts or applications.

## Assignment Components

### assignment5.py File and Expectations

The `assignment5.py` file contains the core component for implementing the Retrieval Augmented Generation strategy. Here's an overview of what's included and what's expected:

#### Key Components

The file contains a `ChatTurnStrategy` class with two main methods:

1. `user_turn_for` method:
   - Already implemented
   - Generates a user turn for the chat session
   - Returns a dictionary with 'role' and 'content' keys

2. `chat_turns_for` method:
   - Needs to be implemented
   - Should generate chat turns, augmenting the response with information from the database

#### Implementation Task

Your main task is to implement the `chat_turns_for` method in the `ChatTurnStrategy` class. This method should:

1. Use the Document Query Model (DQM) to retrieve relevant information based on the user's input.
2. Incorporate the retrieved information into the chat context.
3. Return a list of chat turns that includes both the user's input and the augmented context.

#### Considerations

When implementing the `chat_turns_for` method, consider:

- How many searches to perform using the DQM
- How many documents to retrieve for each search
- Where to inject the retrieved information in the chat turns

#### Reflection Questions

At the end of the file, you'll find several reflection questions. These questions ask you to discuss:

1. The pros and cons of different injection locations for retrieved information
2. Strategies for ensuring retrieved documents are relevant to the conversation context
3. Approaches to structuring prompts that encourage the model to use injected information effectively

You should provide thoughtful answers to these questions, based on your implementation experience and understanding of RAG systems.

#### Expectations

- Complete the implementation of the `chat_turns_for` method
- Ensure your implementation effectively uses the DQM to augment the chat context
- Provide clear and insightful answers to the reflection questions

Remember, the goal is to create a basic RAG system that enhances the AI's responses with relevant information from the document database. Focus on implementing a working solution and reflecting on the design choices and their implications.

## LLM Integration

The `LLMProvider` class in `llm.py` provides a unified interface for different LLM providers, with Google AI Studio as the default. The system supports streaming functionality for chat completions.

## Switching LLM Providers

While Google AI Studio is outlined herre, the system supports other LLM providers. To switch, update the `LLM_PROVIDER` in your `.env` file or when initializing the `ChatConfig`.

Available providers for `LLM_PROVIDER`:

- `ai_studio`: Google AI Studio
- `openai`: OpenAI API
- `groq`: Groq API

Example of switching to OpenAI in your `.env` file:
```
LLM_PROVIDER=openai
API_KEY=your_openai_api_key_here
```

## Configuration

The `ChatConfig` class in `config.py` manages configuration settings for the project, including:
- LLM provider and API settings
- Embedding model selection
- Database path
- Chat parameters (max tokens, temperature, etc.)

## Troubleshooting

If you encounter any issues:

1. Ensure all dependencies are correctly installed (`poetry install`).
2. Check that you're using a compatible Python version (3.9+).
3. Verify that your `.env` file is set up correctly with the necessary API keys.
4. If you're having issues with a specific LLM provider, try switching to a different one in the configuration.

## License

MIT

## Acknowledgments

- Claude Sonnet 3.5 and GPT-4-o1 for documentation and code assistance
- Google AI Studio and Groq for providing free LLM API inference
- Hugging Face for providing embedding models
- Pandas for efficient data handling
- Click for creating the command-line interface
