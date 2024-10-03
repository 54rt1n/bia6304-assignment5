# assignment/assignment5.py

"""
# Assignment 5

Name:
Date:
Tools Used:

## Assignment

The following functions are part of a package for chatting with a document query model backed LLM.

The assignment is to complete the implementations of the missing code.

## Submission

Complete this file and submit it to Canvas, as yourname-assignment5.py

The final implementation should implement the two methods in the strategy class below, and have the questions at the end of the file answered.

"""

from typing import Dict, List

from .config import ChatConfig
from .dqm import DocumentQueryModel

"""
The Strategy Pattern - a GoF design pattern that allows you to define a family of algorithms with a common interface that performss the same task in different ways.
"""

class ChatTurnStrategy:
    def __init__(self, dqm: DocumentQueryModel, config: ChatConfig):
        self.dqm = dqm
        self.config = config

    def user_turn_for(self, user_input: str, history: List[Dict[str, str]] = []) -> Dict[str, str]:
        """
            Generate a user turn for a chat session.
            
            This is what will be stored in the history.
            
            Args:
                user_input (str): The user input.
                history (List[Dict[str, str]]): The chat history.

            Returns:
                Dict[str, str]: The user turn, in the format {"role": "user", "content": user_input}.
        """
        return {"role": "user", "content": user_input}
        
    def chat_turns_for(self, user_input: str, history: List[Dict[str, str]] = []) -> List[Dict[str, str]]:
        """
            Generate a chat session, augmenting the response with information from the database.

            This is what will be passed to the chat complletion API.

            Args:
                user_input (str): The user input.
                history (List[Dict[str, str]]): The chat history.
                
            Returns:
                List[Dict[str, str]]: The chat turns, in the alternating format [{"role": "user", "content": user_input}, {"role": "assistant", "content": assistant_turn}].
        """

        # Part 1:
        # This problem is solvable by performing a query on the dqm, and then injecting the results in to some part of the context.
        #
        # The questions you will need to answer are:
        #   How many searches do you need to make to get the information you need, and how many documents to return for the llm to utilize?
        #
        # How do I place my new information in to the chat stream?
        #  * previous, alternating chat turns - this method, you would craft a fake user and assistant turn, and inject them somewhere before the user input.
        #    be aware, that many models' chat template will error if you do not alternate the roles.
        #  * the user turn - this method, you craft a document and place all of the data directly in to the current user turn
        # 
        # How do I return my data?
        #   Add your history, the rag turn, and the user's turn together to form the chat turns: When you inject the search results
        #   in to the conversation before the request, how close you put the rag information to the user's turn will influence how
        #   much the model will use it in the response, vs prior turn information.

        user_turn = {"role": "user", "content": user_input}

        return history + [user_turn]

"""
## Questions (2-4 paragraphs each)
    
### In the chat_turns_for method, you need to decide where to inject the retrieved information from the document query model. Discuss the pros and cons of different injection locations (e.g., in the user's turn, as a separate assistant turn, or at the beginning of the conversation). How might each approach affect the model's utilization of the injected information?

Your answer here

### The dqm.query method returns results based on vector similarity. How would you ensure that the retrieved documents are not just similar, but relevant to the current conversation context? Propose a strategy to filter or re-rank the query results to improve their relevance to the user's input and the ongoing conversation.

Your answer here

### Designing an effective prompt strategy is crucial for RAG systems. Explain how you would structure the prompt in the chat_turns_for method to encourage the model to use the injected information without blindly copying it. Consider techniques such as few-shot examples, explicit instructions, or dynamic prompt formatting based on the query results.

Your answer here

"""
