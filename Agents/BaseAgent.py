# D:\Jinvexa\Agents\BaseAgent.py

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(self, name: str, llm_client: Optional[Any] = None):
        self.name = name
        self.llm = llm_client
        self.logger = logging.getLogger(f"Agent.{name}")
    
    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input and return output"""
        pass
    
    def _log(self, message: str, level: str = "info"):
        """Log a message"""
        if level == "info":
            self.logger.info(f"[{self.name}] {message}")
        elif level == "error":
            self.logger.error(f"[{self.name}] {message}")
        elif level == "debug":
            self.logger.debug(f"[{self.name}] {message}")