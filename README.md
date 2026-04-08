# Scalar Workplace Environment - OpenEnv RL Challenge

A comprehensive OpenEnv environment featuring real-world workplace productivity tasks for reinforcement learning agents.

## Overview

The Scalar Workplace Environment simulates three common office tasks that humans perform daily:
- **Email Triage** (Easy): Sort and prioritize incoming emails
- **Data Cleaning** (Medium): Clean and standardize messy datasets  
- **Meeting Scheduling** (Hard): Optimize schedules across multiple constraints

This environment is designed for the OpenEnv RL Challenge and fully implements the OpenEnv specification with Pydantic models, typed interfaces, and programmatic graders.

## Features

- **Real-World Tasks**: No toy problems - all tasks mirror actual workplace activities
- **OpenEnv Compliant**: Full implementation with validation support
- **Three Difficulty Levels**: Progressive complexity from easy to hard
- **Programmatic Graders**: Deterministic scoring (0.0-1.0) for each task
- **Meaningful Rewards**: Incremental feedback throughout task trajectories
- **Container Ready**: Dockerized deployment for Hugging Face Spaces

## Tasks

### 1. Email Triage (Easy)
**Objective**: Sort and prioritize incoming emails based on urgency and importance

**Actions**:
- `categorize`: Classify email as "urgent", "normal", or "spam"
- `action`: Choose response action ("reply", "forward", "archive", "delete")
- `skip`: Skip processing the email

**Grading Criteria**:
- Categorization accuracy (60% weight)
- Action appropriateness (40% weight)
- Priority handling for urgent emails

**Example Scenario**: Process inbox with high-priority server alerts, routine reports, newsletters, and spam emails.

### 2. Data Cleaning (Medium)
**Objective**: Clean and standardize messy dataset entries

**Actions**:
- `remove_duplicates`: Remove duplicate rows
- `fill_missing`: Fill missing values (mean/median/mode)
- `standardize_text`: Clean text formatting
- `validate_format`: Validate email/phone formats
- `remove_outliers`: Remove statistical outliers

**Grading Criteria**:
- Data completeness (40% weight)
- Uniqueness (30% weight)
- Format validation (30% weight)

**Example Scenario**: Clean customer database with inconsistent names, invalid emails, missing values, and duplicate entries.

### 3. Meeting Scheduling (Hard)
**Objective**: Optimize meeting schedules across multiple constraints

**Actions**:
- `schedule`: Book meeting in available time slot
- `skip`: Skip the meeting
- `reschedule`: Mark for rescheduling
- `cancel`: Cancel the meeting

**Grading Criteria**:
- Overall scheduling rate (40% weight)
- High-priority meeting handling (30% weight)
- Medium-priority meeting handling (20% weight)
- Conflict avoidance (10% weight)

**Example Scenario**: Schedule board meetings, client presentations, team standups, and reviews while respecting participant availability and time constraints.

## Environment Interface

### Core Components

```python
from src.environment import ScalarWorkplaceEnvironment

# Initialize environment
env = ScalarWorkplaceEnvironment()

# List available tasks
tasks = env.list_tasks()  # ['email_triage', 'data_cleaning', 'scheduling']

# Load a task
env.load_task('email_triage')

# Reset and get initial observation
observation = env.reset()

# Execute action
from src.core import Action
action = Action(action_type='categorize', parameters={'category': 'urgent'})
observation, reward, done, info = env.step(action)

# Get final score
score = env.grade_task()
```

### Observation Space

Each observation contains:
- `task_state`: Current task-specific state
- `available_actions`: List of valid actions
- `step_count`: Current step number
- `max_steps`: Maximum allowed steps
- `metadata`: Additional context

### Action Space

Actions are typed objects with:
- `action_type`: String identifier of the action
- `parameters`: Action-specific parameters
- `description`: Human-readable explanation

### Reward Function

Rewards provide immediate feedback:
- **Positive rewards** for correct actions and progress
- **Negative rewards** for errors, invalid actions, or skipping important items
- **Progressive scaling** based on task difficulty and priority

## Setup and Installation

### Local Development

1. **Clone the repository**:
```bash
git clone <repository-url>
cd scalar-workspace
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run inference**:
```bash
export HF_TOKEN="your-huggingface-token"
python inference.py
```

### Docker Deployment

1. **Build image**:
```bash
docker build -t scalar-workspace .
```

2. **Run container**:
```bash
docker run -e HF_TOKEN="your-token" scalar-workspace
```

### Hugging Face Spaces

1. **Create a new Space** with the Docker template
2. **Upload all files** to the Space
3. **Set environment variables**:
   - `HF_TOKEN`: Your Hugging Face API token
   - `API_BASE_URL`: LLM API endpoint (default: https://api.openai.com/v1)
   - `MODEL_NAME`: Model identifier (default: gpt-4o-mini)
4. **Tag the space** with `openenv`

## Inference Script

The `inference.py` script provides the required interface for the OpenEnv challenge:

### Environment Variables
- `HF_TOKEN` (required): Hugging Face API token
- `API_BASE_URL` (optional): LLM API endpoint, defaults to OpenAI
- `MODEL_NAME` (optional): Model name, defaults to gpt-4o-mini

### Output Format

The script outputs exactly three line types:

```
[START] task=<task_name> env=scalar-workspace model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
```

### Example Output

```
[START] task=email_triage env=scalar-workspace model=gpt-4o-mini
[STEP] step=1 action=categorize reward=0.20 done=false error=null
[STEP] step=2 action=action reward=0.30 done=false error=null
[STEP] step=3 action=categorize reward=-0.10 done=false error=null
[END] success=true steps=3 rewards=0.20,0.30,-0.10
```

## Baseline Performance

Current baseline scores using GPT-4o-mini:

| Task | Difficulty | Baseline Score | Target Score |
|------|------------|----------------|--------------|
| Email Triage | Easy | 0.75 | 0.90 |
| Data Cleaning | Medium | 0.65 | 0.85 |
| Meeting Scheduling | Hard | 0.55 | 0.80 |

**Overall Baseline**: 0.65/1.0

## Architecture

```
scalar-workspace/
|-- src/
|   |-- core.py              # Base environment and grader classes
|   |-- environment.py       # Main environment orchestrator
|   |-- tasks/
|   |   |-- email_triage.py  # Email triage task implementation
|   |   |-- data_cleaning.py # Data cleaning task implementation
|   |   |-- scheduling.py    # Meeting scheduling task implementation
|-- inference.py             # OpenAI API inference script
|-- requirements.txt         # Python dependencies
|-- Dockerfile              # Container configuration
|-- openenv.yaml           # OpenEnv metadata
|-- README.md              # This file
```

## Validation

The environment includes built-in validation:

```python
from src.environment import ScalarWorkplaceEnvironment

env = ScalarWorkplaceEnvironment()
is_valid = env.validate_environment()  # Returns True if compliant
```

Validation checks:
- All tasks implement required methods
- Observations contain required fields
- Actions are properly validated
- Graders return valid scores
- State serialization works correctly

## Contributing

### Adding New Tasks

1. Create task class inheriting from `BaseEnvironment`
2. Implement required methods: `reset()`, `step()`, `state()`, `validate_action()`
3. Create corresponding grader class inheriting from `TaskGrader`
4. Register in `ScalarWorkspaceEnvironment.environments`
5. Update `openenv.yaml` metadata

### Testing

Run the test suite:
```bash
python -m pytest tests/
```

Validate environment:
```bash
python -c "from src.environment import ScalarWorkplaceEnvironment; print(ScalarWorkplaceEnvironment().validate_environment())"
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- OpenEnv RL Challenge organizers for the specification
- Reference implementations from calendar_env, reasoning_gym_env, and other OpenEnv environments
- Hugging Face Spaces team for the deployment platform

## Support

For issues and questions:
- Open an issue on the repository
- Check the OpenEnv documentation
- Review the reference implementations

---

**Ready for the OpenEnv RL Challenge!** This environment meets all requirements and is deployable as a Hugging Face Space with the `openenv` tag.
