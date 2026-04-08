import os
import json
import sys
import traceback
from openai import OpenAI
from src.environment import ScalarWorkplaceEnvironment
from src.core import Action
from flask import Flask, request, jsonify

app = Flask(__name__)

# Read environment variables with defaults where required
# Force correct router URL since api-inference.huggingface.co is deprecated
API_BASE_URL = "https://router.huggingface.co/v1"
# Force correct Hugging Face model
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("Token_Hf")

if HF_TOKEN is None:
    print("ERROR: HF_TOKEN environment variable is required")
    print("Please set HF_TOKEN in Space settings")
    print("Available environment variables:", list(os.environ.keys()))
    # Exit gracefully instead of using dummy token
    sys.exit(1)

# Initialize OpenAI client
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

def format_observation_for_prompt(observation) -> str:
    """Format observation for LLM prompt"""
    obs_dict = {
        "task_state": observation.task_state,
        "available_actions": observation.available_actions,
        "step_count": observation.step_count,
        "max_steps": observation.max_steps
    }
    return json.dumps(obs_dict, indent=2, default=str)

def parse_action_from_response(response_text: str, available_actions: list) -> Action:
    """Parse LLM response into Action object"""
    try:
        # Try to extract JSON from response
        response_text = response_text.strip()
        
        # Look for JSON block
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            json_text = response_text[start:end].strip()
        elif "{" in response_text and "}" in response_text:
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            json_text = response_text[start:end]
        else:
            # Fallback: try to parse the whole response as JSON
            json_text = response_text
        
        action_data = json.loads(json_text)
        
        # Validate action type
        action_type = action_data.get("action_type", "")
        if action_type not in available_actions:
            # Default to first available action if invalid
            action_type = available_actions[0] if available_actions else "skip"
        
        return Action(
            action_type=action_type,
            parameters=action_data.get("parameters", {}),
            description=action_data.get("description", "")
        )
    
    except Exception:
        # Fallback action
        return Action(
            action_type=available_actions[0] if available_actions else "skip",
            parameters={},
            description="Fallback action due to parsing error"
        )

def create_task_prompt(task_name: str, observation, task_info: dict) -> str:
    """Create prompt for the LLM"""
    
    base_prompt = f"""You are an AI assistant working on a {task_info.get('difficulty', 'unknown')} difficulty task: {task_info.get('description', 'Unknown task')}.

Current task state:
{format_observation_for_prompt(observation)}

Available actions: {', '.join(observation.available_actions)}

Your goal is to complete this task successfully by choosing appropriate actions at each step.

Respond with a JSON object containing:
- action_type: one of the available actions
- parameters: object with any required parameters for the action
- description: brief description of what you're doing

Example response:
```json
{{
    "action_type": "categorize",
    "parameters": {{
        "category": "urgent"
    }},
    "description": "Mark this email as urgent due to critical content"
}}
```

Choose your action for this step:"""
    
    # Add task-specific guidance
    if task_name == "email_triage":
        base_prompt += """

For email triage:
- categorize: Set category to "urgent", "normal", or "spam"
- action: Set action to "reply", "forward", "archive", or "delete"
- skip: Skip this email (not recommended for urgent emails)

Consider email urgency, sender importance, and content relevance."""
    
    elif task_name == "data_cleaning":
        base_prompt += """

For data cleaning:
- remove_duplicates: Remove duplicate rows from the dataset
- fill_missing: Fill missing values (specify column and method: "mean", "median", or "mode")
- standardize_text: Clean text formatting (specify column)
- validate_format: Validate data format (specify column and format: "email" or "phone")
- remove_outliers: Remove statistical outliers (specify column)

Focus on improving data quality metrics shown in the task state."""
    
    elif task_name == "scheduling":
        base_prompt += """

For scheduling:
- schedule: Schedule meeting (specify time_slot_id from available options)
- skip: Skip this meeting (not recommended for high priority)
- reschedule: Mark for rescheduling
- cancel: Cancel the meeting

Consider meeting priority, participant availability, and time constraints."""
    
    return base_prompt

def run_inference():
    """Main inference function"""
    # Initialize environment
    env = ScalarWorkplaceEnvironment()
    
    # Get available tasks
    tasks = env.list_tasks()
    
    # Run inference for each task
    for task_name in tasks:
        try:
            # Load task
            env.load_task(task_name)
            task_info = env.get_task_info(task_name)
            
            # Reset environment
            observation = env.reset()
            
            # Print start line
            print(f"[START] task={task_name} env=scalar-workspace model={MODEL_NAME}")
            
            rewards = []
            step_count = 0
            success = False
            last_error = None
            
            # Run episode
            while step_count < observation.max_steps and not getattr(observation, 'done', False):
                step_count += 1
                
                try:
                    # Create prompt
                    prompt = create_task_prompt(task_name, observation, task_info)
                    
                    # Get LLM response
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": "You are a helpful AI assistant completing workplace productivity tasks."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=500,
                        temperature=0.7
                    )
                    
                    response_text = response.choices[0].message.content
                    
                    # Parse action
                    action = parse_action_from_response(response_text, observation.available_actions)
                    
                    # Execute action
                    observation, reward, done, info = env.step(action)
                    rewards.append(reward.value)
                    
                    # Print step line
                    print(f"[STEP] step={step_count} action={action.action_type} reward={reward.value:.2f} done={done} error={last_error or 'null'}")
                    
                    if done:
                        success = True
                        break
                    
                except Exception as e:
                    last_error = str(e)
                    rewards.append(-0.1)  # Penalty for errors
                    print(f"[STEP] step={step_count} action=error reward=-0.10 done=false error={last_error}")
                    
                    # Try to continue with a default action
                    try:
                        default_action = Action(action_type="skip", parameters={})
                        observation, reward, done, info = env.step(default_action)
                        if done:
                            success = True
                            break
                    except:
                        break
            
            # Grade the task
            try:
                final_score = env.grade_task()
                if final_score >= 0.8:
                    success = True
            except:
                final_score = 0.0
            
            # Print end line
            rewards_str = ",".join([f"{r:.2f}" for r in rewards])
            print(f"[END] success={success} steps={step_count} rewards={rewards_str}")
            
            # Close environment
            env.close()
            
        except Exception as e:
            print(f"[START] task={task_name} env=scalar-workspace model={MODEL_NAME}")
            print(f"[STEP] step=1 action=error reward=-0.10 done=false error={str(e)}")
            print(f"[END] success=false steps=1 rewards=-0.10")
            
            # Try to close environment
            try:
                env.close()
            except:
                pass

@app.post("/reset")
def reset():
    """Reset environment for OpenEnv compatibility"""
    try:
        # Reset the environment state
        global env
        env = ScalarWorkplaceEnvironment()
        return jsonify({"status": "success", "message": "Environment reset successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    run_inference()
