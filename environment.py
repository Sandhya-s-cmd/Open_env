from typing import Dict, Any, List, Optional
from .core import BaseEnvironment, TaskGrader
from .tasks.email_triage import EmailTriageEnvironment, EmailTriageGrader
from .tasks.data_cleaning import DataCleaningEnvironment, DataCleaningGrader
from .tasks.scheduling import SchedulingEnvironment, SchedulingGrader


class ScalarWorkplaceEnvironment:
    """Main environment class for Scalar Workplace tasks"""
    
    def __init__(self):
        self.environments = {
            "email_triage": (EmailTriageEnvironment, EmailTriageGrader),
            "data_cleaning": (DataCleaningEnvironment, DataCleaningGrader),
            "scheduling": (SchedulingEnvironment, SchedulingGrader)
        }
        self.current_env = None
        self.current_task = None
        self.current_grader = None
        
    def list_tasks(self) -> List[str]:
        """List all available tasks"""
        return list(self.environments.keys())
    
    def get_task_info(self, task_name: str) -> Dict[str, Any]:
        """Get information about a specific task"""
        if task_name not in self.environments:
            return {}
        
        env_class, grader_class = self.environments[task_name]
        temp_env = env_class()
        
        return {
            "name": task_name,
            "difficulty": temp_env.difficulty,
            "max_steps": temp_env.max_steps,
            "description": self._get_task_description(task_name)
        }
    
    def load_task(self, task_name: str, **kwargs) -> bool:
        """Load a specific task"""
        if task_name not in self.environments:
            return False
        
        env_class, grader_class = self.environments[task_name]
        self.current_env = env_class(**kwargs)
        self.current_task = task_name
        self.current_grader = grader_class()
        
        return True
    
    def reset(self):
        """Reset the current environment"""
        if self.current_env is None:
            raise ValueError("No task loaded. Call load_task() first.")
        
        return self.current_env.reset()
    
    def step(self, action):
        """Execute a step in the current environment"""
        if self.current_env is None:
            raise ValueError("No task loaded. Call load_task() first.")
        
        return self.current_env.step(action)
    
    def state(self) -> Dict[str, Any]:
        """Get current environment state"""
        if self.current_env is None:
            raise ValueError("No task loaded. Call load_task() first.")
        
        return self.current_env.state()
    
    def grade_task(self) -> float:
        """Grade the completed task"""
        if self.current_env is None or self.current_grader is None:
            raise ValueError("No task loaded. Call load_task() first.")
        
        final_state = self.state()
        return self.current_grader.grade(final_state)
    
    def get_grading_criteria(self) -> Dict[str, Any]:
        """Get grading criteria for current task"""
        if self.current_grader is None:
            raise ValueError("No task loaded. Call load_task() first.")
        
        return self.current_grader.get_grading_criteria()
    
    def close(self):
        """Close the environment"""
        if self.current_env:
            self.current_env.close()
    
    def _get_task_description(self, task_name: str) -> str:
        """Get description for a task"""
        descriptions = {
            "email_triage": "Sort and prioritize incoming emails based on urgency and importance",
            "data_cleaning": "Clean and standardize messy dataset entries",
            "scheduling": "Optimize meeting schedules across multiple constraints"
        }
        return descriptions.get(task_name, "Unknown task")
    
    def validate_environment(self) -> bool:
        """Validate that the environment meets OpenEnv specifications"""
        try:
            # Test all tasks
            for task_name in self.list_tasks():
                if not self.load_task(task_name):
                    return False
                
                # Test reset
                obs = self.reset()
                if not hasattr(obs, 'task_state') or not hasattr(obs, 'available_actions'):
                    return False
                
                # Test step
                from .core import Action
                action = Action(action_type="test")
                try:
                    self.step(action)
                except:
                    # Some actions might be invalid, that's ok
                    pass
                
                # Test state
                state = self.state()
                if not isinstance(state, dict):
                    return False
                
                # Test grader
                criteria = self.get_grading_criteria()
                if not isinstance(criteria, dict):
                    return False
            
            return True
        except Exception as e:
            print(f"Validation error: {e}")
            return False
