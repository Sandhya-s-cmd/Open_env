from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field
import json


class Observation(BaseModel):
    """Base observation model for OpenEnv environments"""
    task_state: Dict[str, Any] = Field(description="Current state of the task")
    available_actions: List[str] = Field(description="List of available actions")
    step_count: int = Field(description="Current step number")
    max_steps: int = Field(description="Maximum allowed steps")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class Action(BaseModel):
    """Base action model for OpenEnv environments"""
    action_type: str = Field(description="Type of action to perform")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Action parameters")
    description: Optional[str] = Field(None, description="Human-readable description")


class Reward(BaseModel):
    """Base reward model for OpenEnv environments"""
    value: float = Field(description="Reward value between -1.0 and 1.0")
    reason: Optional[str] = Field(None, description="Reason for the reward")
    progress: float = Field(default=0.0, description="Progress towards goal (0.0 to 1.0)")


class Info(BaseModel):
    """Additional information returned by environment step"""
    task_complete: bool = Field(default=False, description="Whether the task is complete")
    score: float = Field(default=0.0, description="Current task score")
    errors: List[str] = Field(default_factory=list, description="Any errors that occurred")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional info")


class BaseEnvironment(ABC):
    """Base class for OpenEnv environments"""
    
    def __init__(self, max_steps: int = 50):
        self.max_steps = max_steps
        self.current_step = 0
        self.task_name = ""
        self.difficulty = ""
        
    @abstractmethod
    def reset(self) -> Observation:
        """Reset the environment and return initial observation"""
        pass
    
    @abstractmethod
    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Info]:
        """Execute one step in the environment"""
        pass
    
    @abstractmethod
    def state(self) -> Dict[str, Any]:
        """Return current environment state"""
        pass
    
    @abstractmethod
    def validate_action(self, action: Action) -> bool:
        """Validate if action is valid in current state"""
        pass
    
    def get_available_actions(self) -> List[str]:
        """Get list of available action types"""
        return []
    
    def close(self):
        """Clean up resources"""
        pass


class TaskGrader(ABC):
    """Base class for task graders"""
    
    @abstractmethod
    def grade(self, final_state: Dict[str, Any]) -> float:
        """Grade task completion and return score between 0.0 and 1.0"""
        pass
    
    @abstractmethod
    def get_grading_criteria(self) -> Dict[str, Any]:
        """Return grading criteria description"""
        pass
