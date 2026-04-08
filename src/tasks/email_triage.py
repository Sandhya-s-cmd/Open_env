import random
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
from ..core import BaseEnvironment, Observation, Action, Reward, Info, TaskGrader


class Email:
    """Email model for triage task"""
    
    def __init__(self, subject: str, sender: str, body: str, urgency: str, 
                 received_time: datetime, email_id: str):
        self.subject = subject
        self.sender = sender
        self.body = body
        self.urgency = urgency  # high, medium, low
        self.received_time = received_time
        self.email_id = email_id
        self.category = None  # urgent, normal, spam
        self.action_taken = None  # reply, forward, archive, delete


class EmailTriageEnvironment(BaseEnvironment):
    """Email triage environment - easy difficulty"""
    
    def __init__(self, max_steps: int = 20):
        super().__init__(max_steps)
        self.task_name = "email_triage"
        self.difficulty = "easy"
        self.emails = []
        self.processed_emails = []
        self.current_email_index = 0
        
    def reset(self) -> Observation:
        """Reset environment with new email batch"""
        self.current_step = 0
        self.emails = self._generate_email_batch()
        self.processed_emails = []
        self.current_email_index = 0
        
        return Observation(
            task_state={
                "current_email": self.emails[0].__dict__ if self.emails else None,
                "remaining_emails": len(self.emails),
                "processed_count": 0,
                "inbox_summary": self._get_inbox_summary()
            },
            available_actions=self.get_available_actions(),
            step_count=self.current_step,
            max_steps=self.max_steps,
            metadata={"task": "email_triage", "difficulty": "easy"}
        )
    
    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Info]:
        """Process email triage action"""
        self.current_step += 1
        
        if not self.validate_action(action):
            return self._get_current_observation(), Reward(value=-0.1, reason="Invalid action"), False, \
                   Info(errors=["Invalid action"])
        
        current_email = self.emails[self.current_email_index]
        reward_value, reward_reason = self._calculate_reward(action, current_email)
        
        # Apply action to current email
        self._apply_action(action, current_email)
        self.processed_emails.append(current_email)
        
        # Move to next email or finish
        done = False
        if self.current_email_index < len(self.emails) - 1:
            self.current_email_index += 1
        else:
            done = True
        
        observation = self._get_current_observation()
        info = Info(
            task_complete=done,
            score=self._calculate_score(),
            metadata={"last_action": action.action_type}
        )
        
        return observation, Reward(value=reward_value, reason=reward_reason), done, info
    
    def state(self) -> Dict[str, Any]:
        """Return current environment state"""
        return {
            "emails": [email.__dict__ for email in self.emails],
            "processed_emails": [email.__dict__ for email in self.processed_emails],
            "current_email_index": self.current_email_index,
            "step_count": self.current_step,
            "max_steps": self.max_steps
        }
    
    def validate_action(self, action: Action) -> bool:
        """Validate email triage action"""
        valid_actions = {"categorize", "action", "skip"}
        return action.action_type in valid_actions
    
    def get_available_actions(self) -> List[str]:
        """Get available actions"""
        return ["categorize", "action", "skip"]
    
    def _generate_email_batch(self) -> List[Email]:
        """Generate a batch of emails for triage"""
        emails = []
        
        # High urgency emails
        emails.append(Email(
            subject="URGENT: Server Down - Production Issues",
            sender="alerts@company.com",
            body="Critical system failure detected. Immediate attention required.",
            urgency="high",
            received_time=datetime.now() - timedelta(minutes=5),
            email_id="email_001"
        ))
        
        emails.append(Email(
            subject="Meeting Reschedule Request - CEO",
            sender="ceo@company.com",
            body="Need to reschedule tomorrow's board meeting due to conflict.",
            urgency="high",
            received_time=datetime.now() - timedelta(hours=1),
            email_id="email_002"
        ))
        
        # Medium urgency emails
        emails.append(Email(
            subject="Weekly Report Submission",
            sender="manager@company.com",
            body="Please submit your weekly report by end of day.",
            urgency="medium",
            received_time=datetime.now() - timedelta(hours=3),
            email_id="email_003"
        ))
        
        emails.append(Email(
            subject="Team Lunch Planning",
            sender="hr@company.com",
            body="Planning team lunch for next Friday. Please vote on preferences.",
            urgency="medium",
            received_time=datetime.now() - timedelta(hours=5),
            email_id="email_004"
        ))
        
        # Low urgency emails
        emails.append(Email(
            subject="Newsletter: Industry Updates",
            sender="newsletter@industry.com",
            body="Latest industry news and updates from this week.",
            urgency="low",
            received_time=datetime.now() - timedelta(days=1),
            email_id="email_005"
        ))
        
        # Spam
        emails.append(Email(
            subject="You Won $1,000,000!!!",
            sender="scam@fake.com",
            body="Congratulations! Click here to claim your prize!",
            urgency="low",
            received_time=datetime.now() - timedelta(days=2),
            email_id="email_006"
        ))
        
        random.shuffle(emails)
        return emails
    
    def _calculate_reward(self, action: Action, email: Email) -> Tuple[float, str]:
        """Calculate reward for action on email"""
        if action.action_type == "categorize":
            category = action.parameters.get("category", "")
            expected_category = self._get_expected_category(email)
            
            if category == expected_category:
                return 0.2, f"Correctly categorized as {category}"
            else:
                return -0.1, f"Incorrect categorization: expected {expected_category}, got {category}"
        
        elif action.action_type == "action":
            email_action = action.parameters.get("action", "")
            expected_action = self._get_expected_action(email)
            
            if email_action == expected_action:
                return 0.3, f"Appropriate action: {email_action}"
            else:
                return -0.15, f"Inappropriate action: expected {expected_action}, got {email_action}"
        
        elif action.action_type == "skip":
            return -0.05, "Skipped email without processing"
        
        return 0.0, "No reward"
    
    def _get_expected_category(self, email: Email) -> str:
        """Get expected category for email"""
        if email.urgency == "high" and "alert" in email.sender.lower():
            return "urgent"
        elif email.urgency == "high":
            return "urgent"
        elif "scam" in email.sender.lower() or "won" in email.subject.lower():
            return "spam"
        else:
            return "normal"
    
    def _get_expected_action(self, email: Email) -> str:
        """Get expected action for email"""
        if email.urgency == "high":
            return "reply"
        elif "scam" in email.sender.lower():
            return "delete"
        elif "newsletter" in email.sender.lower():
            return "archive"
        else:
            return "reply"
    
    def _apply_action(self, action: Action, email: Email):
        """Apply action to email"""
        if action.action_type == "categorize":
            email.category = action.parameters.get("category", "")
        elif action.action_type == "action":
            email.action_taken = action.parameters.get("action", "")
    
    def _get_current_observation(self) -> Observation:
        """Get current observation"""
        current_email = None
        if self.current_email_index < len(self.emails):
            current_email = self.emails[self.current_email_index].__dict__
        
        return Observation(
            task_state={
                "current_email": current_email,
                "remaining_emails": len(self.emails) - self.current_email_index - 1,
                "processed_count": len(self.processed_emails),
                "inbox_summary": self._get_inbox_summary()
            },
            available_actions=self.get_available_actions(),
            step_count=self.current_step,
            max_steps=self.max_steps,
            metadata={"task": "email_triage", "difficulty": "easy"}
        )
    
    def _get_inbox_summary(self) -> Dict[str, Any]:
        """Get summary of inbox state"""
        categories = {"urgent": 0, "normal": 0, "spam": 0}
        for email in self.emails:
            if email.category:
                categories[email.category] += 1
        
        return {
            "total_emails": len(self.emails),
            "categories": categories,
            "processed": len(self.processed_emails)
        }
    
    def _calculate_score(self) -> float:
        """Calculate overall task score"""
        if not self.processed_emails:
            return 0.0
        
        correct_categorizations = 0
        correct_actions = 0
        
        for email in self.processed_emails:
            expected_category = self._get_expected_category(email)
            expected_action = self._get_expected_action(email)
            
            if email.category == expected_category:
                correct_categorizations += 1
            if email.action_taken == expected_action:
                correct_actions += 1
        
        category_score = correct_categorizations / len(self.processed_emails)
        action_score = correct_actions / len(self.processed_emails)
        
        return (category_score + action_score) / 2


class EmailTriageGrader(TaskGrader):
    """Grader for email triage task"""
    
    def grade(self, final_state: Dict[str, Any]) -> float:
        """Grade email triage performance"""
        processed_emails = final_state.get("processed_emails", [])
        
        if not processed_emails:
            return 0.0
        
        correct_categorizations = 0
        correct_actions = 0
        total_emails = len(processed_emails)
        
        for email_data in processed_emails:
            email = Email(**email_data)
            
            # Check categorization
            expected_category = self._get_expected_category(email)
            if email.category == expected_category:
                correct_categorizations += 1
            
            # Check actions
            expected_action = self._get_expected_action(email)
            if email.action_taken == expected_action:
                correct_actions += 1
        
        category_score = correct_categorizations / total_emails
        action_score = correct_actions / total_emails
        
        # Weight categorization more heavily
        final_score = (category_score * 0.6 + action_score * 0.4)
        
        return min(1.0, max(0.0, final_score))
    
    def get_grading_criteria(self) -> Dict[str, Any]:
        """Return grading criteria"""
        return {
            "categorization_weight": 0.6,
            "action_weight": 0.4,
            "categories": ["urgent", "normal", "spam"],
            "actions": ["reply", "forward", "archive", "delete"],
            "scoring": "Weighted average of correct categorizations and actions"
        }
    
    def _get_expected_category(self, email: Email) -> str:
        """Get expected category for email"""
        if email.urgency == "high" and "alert" in email.sender.lower():
            return "urgent"
        elif email.urgency == "high":
            return "urgent"
        elif "scam" in email.sender.lower() or "won" in email.subject.lower():
            return "spam"
        else:
            return "normal"
    
    def _get_expected_action(self, email: Email) -> str:
        """Get expected action for email"""
        if email.urgency == "high":
            return "reply"
        elif "scam" in email.sender.lower():
            return "delete"
        elif "newsletter" in email.sender.lower():
            return "archive"
        else:
            return "reply"
