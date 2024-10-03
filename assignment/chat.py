# assignment/chat.py

from typing import List, Dict, Callable, Tuple, Optional

from .assignment5 import ChatTurnStrategy
from .config  import ChatConfig
from .llm import LLMProvider
from .dqm import DocumentQueryModel

HELP = """Commands:
- (b)ack: Go back to the previous message
- (h)elp: Show this help message
- new: Start a new chat
- (p)rompt <message>: Update the system message
- re(d)raw: Redraw the screen
- (r)etry: Retry the previous user input
- (s)earch <query>: Search your documents
- top <n>: Set the top n results to use
- temp <n>: Set the temperature
- (q)uit/exit: End the chat
"""


class ChatManager:
    def __init__(self, llm: LLMProvider, dqm: DocumentQueryModel, config: ChatConfig, clear_output=Callable[[], None]):
        self.llm = llm
        self.dqm = dqm
        self.config = config
        self.clear_output = clear_output
        self.chat_strategy = ChatTurnStrategy(dqm, config)

        self.running = False
        self.history : List[Dict[str, str]] = []

    def render_conversation(self, messages: List[Dict[str, str]]) -> None:
        if self.clear_output is not None:
            self.clear_output()
        for message in messages:
            rolename = message['role'].capitalize()
            print(f"{rolename}: {message['content']}\n")
        print(flush=True)

    def add_history(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})

    def clear_history(self) -> None:
        self.history.clear()

    def run_once(self) -> Tuple[str, Optional[str]]:
        # Each iteration, we render the conversation
        self.render_conversation(self.history)

        # Next, we get the user input, and handle special commands
        user_input = input("You (h for help): ").strip()

        # If the user input is empty, we skip this iteration
        if not user_input or user_input.strip() == '':
            print("Type 'h' for help, 'q' to quit.")
            return 'pass', None

        # Handle single word commands
        lowered = user_input.lower()
        if lowered in ['q', 'quit', 'exit']:
            return 'quit', None
        elif lowered in ['r', 'retry']:
            if len(self.history) >= 2:
                user_input = self.history[-2]["content"]
                self.history = self.history[:-2]
        elif lowered in ['b', 'back']:
            if len(self.history) >= 2:
                self.history = self.history[:-2]
            return 'back', 'Back one turn'
        elif lowered == 'new':
            self.clear_history()
            return 'new', 'New chat started'
        elif lowered == 'redraw':
            return 'redraw', 'Redrew the screen'
        elif lowered in ['h', 'help']:
            return 'help', None

        # Handle multi-word commands
        multi = lowered.split()
        # 'prompt' allows the user to update the system message
        if len(multi) > 1 and multi[0] in ['p', 'prompt']:
            self.config.system_message = ' '.join(multi[1:])
            return 'Prompt Updated'
        # Set our top-n
        elif len(multi) > 1 and multi[0] in ['top']:
            try:
                intval = int(multi[1])
                if intval > 0:
                    self.config.top_n = intval
                    return 'top_n_set', f'Top N set to {self.config.top_n}'
                else:
                    return 'invalid_top_n', 'Invalid top N value'
            except ValueError:
                return 'invalid_top_n', 'Invalid top N value'
        # Set our temperature
        elif len(multi) > 1 and multi[0] in ['temp']:
            try:
                self.config.temperature = float(multi[1])
                return 'temperature_set', f'Temperature set to {self.config.temperature}'
            except ValueError:
                return 'invalid_temperature', 'Invalid temperature value'
        # Search for documents
        elif len(multi) > 1 and multi[0] in ['s', 'search']:
            documents = self.dqm.query(' '.join(multi[1:]))
            if documents.empty:
                return 'no_results', 'No documents found'
            else:
                for i, row in documents.reset_index().iterrows():
                    id = row['doc_id']
                    dist = row['distance']
                    print(f"Document {id} (distance: {dist:.2f})")
                    print()
                return 'found_document', None

        user_turn = self.chat_strategy.user_turn_for(user_input)
        chat_turns = self.chat_strategy.chat_turns_for(user_input, self.history)

        if self.config.debug:
            self.render_conversation(chat_turns)

        print(f"Assistant: ", end='', flush=True)

        chunks = []

        for t in self.llm.stream_turns(chat_turns, self.config):
            if t is not None:
                print(t, end='', flush=True)
                chunks.append(t)
            else:
                print('', flush=True)

        response = ''.join(chunks)

        self.add_history(**user_turn)
        self.add_history("assistant", response)

        return 'continue', None

    def chat_loop(self) -> None:
        self.running = True
        while self.running:
            try:
                result, message = self.run_once()
                
                enter = True
                if result == 'quit':
                    self.running = False
                    enter = False
                elif result == 'redraw':
                    enter = False
                elif result == 'help':
                    print()
                    print(HELP)
                elif result == 'continue':
                    enter = False
                else:
                    if message is not None:
                        print(f"{result}: {message}")
                    else:
                        print(f"{result}", end='')

                if enter:
                    print()
                    input("Hit enter to continue...")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"An error occurred: {e}")

                print()
                input("Hit enter to continue...")
            
        self.running = False
        print("Chat session ended.")

    def __repr__(self):
        return f"ChatManager(history={len(self.history)} documents={self.dqm.document_count} config={self.config})"