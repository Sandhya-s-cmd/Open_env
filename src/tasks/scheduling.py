import random
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
from ..core import BaseEnvironment, Observation, Action, Reward, Info, TaskGrader


class Meeting:
    """Meeting model for scheduling task"""
    
    def __init__(self, title: str, duration: int, priority: str, participants: List[str], 
                 constraints: Dict[str, Any], meeting_id: str):
        self.title = title
        self.duration = duration  # in minutes
        self.priority = priority  # high, medium, low
        self.participants = participants
        self.constraints = constraints
        self.meeting_id = meeting_id
        self.scheduled_time = None
        self.status = "unscheduled"  # unscheduled, scheduled, conflict


class TimeSlot:
    """Time slot model"""
    
    def __init__(self, start_time: datetime, end_time: datetime):
        self.start_time = start_time
        self.end_time = end_time
        self.occupied = False
        self.meeting = None


class SchedulingEnvironment(BaseEnvironment):
    """Meeting scheduling environment - hard difficulty"""
    
    def __init__(self, max_steps: int = 40):
        super().__init__(max_steps)
        self.task_name = "scheduling"
        self.difficulty = "hard"
        self.meetings = []
        self.time_slots = []
        self.participant_availability = {}
        self.current_meeting_index = 0
        self.schedule_conflicts = []
        
    def reset(self) -> Observation:
        """Reset environment with scheduling challenge"""
        self.current_step = 0
        self.meetings = self._generate_meetings()
        self.time_slots = self._generate_time_slots()
        self.participant_availability = self._generate_participant_availability()
        self.current_meeting_index = 0
        self.schedule_conflicts = []
        
        return Observation(
            task_state={
                "current_meeting": self.meetings[0].__dict__ if self.meetings else None,
                "meetings_remaining": len(self.meetings),
                "scheduled_meetings": self._get_scheduled_meetings(),
                "time_slots_available": self._get_available_time_slots(),
                "participant_availability": self.participant_availability,
                "schedule_conflicts": len(self.schedule_conflicts)
            },
            available_actions=self.get_available_actions(),
            step_count=self.current_step,
            max_steps=self.max_steps,
            metadata={"task": "scheduling", "difficulty": "hard"}
        )
    
    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Info]:
        """Process scheduling action"""
        self.current_step += 1
        
        if not self.validate_action(action):
            return self._get_current_observation(), Reward(value=-0.1, reason="Invalid action"), False, \
                   Info(errors=["Invalid action"])
        
        current_meeting = self.meetings[self.current_meeting_index]
        reward_value, reward_reason = self._calculate_reward(action, current_meeting)
        
        # Apply scheduling action
        action_result = self._apply_scheduling_action(action, current_meeting)
        
        # Check for conflicts
        self._check_schedule_conflicts()
        
        # Move to next meeting or finish
        done = False
        if self.current_meeting_index < len(self.meetings) - 1:
            self.current_meeting_index += 1
        else:
            done = True
        
        observation = self._get_current_observation()
        info = Info(
            task_complete=done,
            score=self._calculate_schedule_score(),
            metadata={
                "last_action": action.action_type,
                "conflicts": len(self.schedule_conflicts),
                "scheduled_count": len(self._get_scheduled_meetings())
            }
        )
        
        return observation, Reward(value=reward_value, reason=reward_reason), done, info
    
    def state(self) -> Dict[str, Any]:
        """Return current environment state"""
        return {
            "meetings": [meeting.__dict__ for meeting in self.meetings],
            "time_slots": [slot.__dict__ for slot in self.time_slots],
            "participant_availability": self.participant_availability,
            "current_meeting_index": self.current_meeting_index,
            "schedule_conflicts": self.schedule_conflicts,
            "step_count": self.current_step,
            "max_steps": self.max_steps
        }
    
    def validate_action(self, action: Action) -> bool:
        """Validate scheduling action"""
        valid_actions = {"schedule", "skip", "reschedule", "cancel"}
        return action.action_type in valid_actions
    
    def get_available_actions(self) -> List[str]:
        """Get available scheduling actions"""
        return ["schedule", "skip", "reschedule", "cancel"]
    
    def _generate_meetings(self) -> List[Meeting]:
        """Generate meetings to be scheduled"""
        meetings = []
        
        # High priority meetings
        meetings.append(Meeting(
            title="Board Meeting",
            duration=120,
            priority="high",
            participants=["CEO", "CFO", "CTO", "COO"],
            constraints={"must_be_before": "17:00", "prefer_morning": True},
            meeting_id="meeting_001"
        ))
        
        meetings.append(Meeting(
            title="Client Presentation",
            duration=90,
            priority="high",
            participants=["Sales Lead", "Technical Lead", "CEO"],
            constraints={"must_be_after": "10:00", "prefer_afternoon": False},
            meeting_id="meeting_002"
        ))
        
        # Medium priority meetings
        meetings.append(Meeting(
            title="Team Standup",
            duration=30,
            priority="medium",
            participants=["Dev Team"],
            constraints={"prefer_morning": True, "recurring": True},
            meeting_id="meeting_003"
        ))
        
        meetings.append(Meeting(
            title="Design Review",
            duration=60,
            priority="medium",
            participants=["Design Lead", "Product Manager", "Dev Lead"],
            constraints={"prefer_afternoon": True},
            meeting_id="meeting_004"
        ))
        
        meetings.append(Meeting(
            title="Budget Review",
            duration=90,
            priority="medium",
            participants=["CFO", "Finance Team", "CEO"],
            constraints={"must_be_before": "16:00"},
            meeting_id="meeting_005"
        ))
        
        # Low priority meetings
        meetings.append(Meeting(
            title="Team Building",
            duration=60,
            priority="low",
            participants=["All Staff"],
            constraints={"prefer_friday": True, "flexible": True},
            meeting_id="meeting_006"
        ))
        
        meetings.append(Meeting(
            title="Training Session",
            duration=120,
            priority="low",
            participants=["New Hires", "HR"],
            constraints={"prefer_morning": True, "flexible": True},
            meeting_id="meeting_007"
        ))
        
        random.shuffle(meetings)
        return meetings
    
    def _generate_time_slots(self) -> List[TimeSlot]:
        """Generate available time slots for the week"""
        time_slots = []
        base_date = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
        
        # Generate slots for 5 days (Monday-Friday)
        for day in range(5):
            current_date = base_date + timedelta(days=day)
            
            # Generate hourly slots from 9 AM to 5 PM
            for hour in range(8):  # 9 AM to 5 PM (8 hours)
                start_time = current_date + timedelta(hours=hour)
                end_time = start_time + timedelta(hours=1)
                time_slots.append(TimeSlot(start_time, end_time))
        
        return time_slots
    
    def _generate_participant_availability(self) -> Dict[str, List[str]]:
        """Generate participant availability constraints"""
        availability = {}
        
        # Define busy times for key participants
        availability["CEO"] = [
            "Monday 14:00-15:00",  # Existing meeting
            "Tuesday 10:00-11:00",
            "Wednesday 15:00-16:00"
        ]
        
        availability["CFO"] = [
            "Monday 10:00-12:00",  # Financial review
            "Thursday 14:00-16:00"
        ]
        
        availability["CTO"] = [
            "Tuesday 09:00-11:00",  # Tech planning
            "Friday 13:00-15:00"
        ]
        
        availability["Sales Lead"] = [
            "Wednesday 10:00-12:00",  # Client calls
            "Friday 09:00-10:00"
        ]
        
        return availability
    
    def _calculate_reward(self, action: Action, meeting: Meeting) -> Tuple[float, str]:
        """Calculate reward for scheduling action"""
        if action.action_type == "schedule":
            time_slot_id = action.parameters.get("time_slot_id", "")
            time_slot = self._find_time_slot(time_slot_id)
            
            if not time_slot:
                return -0.2, "Invalid time slot"
            
            # Check if scheduling respects constraints
            constraint_score = self._check_constraints(meeting, time_slot)
            priority_bonus = 0.1 if meeting.priority == "high" else 0.05 if meeting.priority == "medium" else 0.0
            
            return constraint_score + priority_bonus, f"Scheduled with constraint score: {constraint_score:.2f}"
        
        elif action.action_type == "skip":
            penalty = -0.1 if meeting.priority == "high" else -0.05 if meeting.priority == "medium" else -0.02
            return penalty, f"Skipped {meeting.priority} priority meeting"
        
        elif action.action_type == "reschedule":
            return 0.0, "Rescheduled meeting"
        
        elif action.action_type == "cancel":
            penalty = -0.15 if meeting.priority == "high" else -0.08 if meeting.priority == "medium" else -0.03
            return penalty, f"Cancelled {meeting.priority} priority meeting"
        
        return 0.0, "No reward"
    
    def _check_constraints(self, meeting: Meeting, time_slot: TimeSlot) -> float:
        """Check if meeting respects constraints"""
        score = 0.0
        
        # Check time constraints
        start_hour = time_slot.start_time.hour
        
        if "must_be_before" in meeting.constraints:
            before_time = meeting.constraints["must_be_before"]
            before_hour = int(before_time.split(":")[0])
            if start_hour < before_hour:
                score += 0.3
            else:
                score -= 0.2
        
        if "must_be_after" in meeting.constraints:
            after_time = meeting.constraints["must_be_after"]
            after_hour = int(after_time.split(":")[0])
            if start_hour >= after_hour:
                score += 0.3
            else:
                score -= 0.2
        
        # Check preference constraints
        if "prefer_morning" in meeting.constraints and meeting.constraints["prefer_morning"]:
            if 9 <= start_hour < 12:
                score += 0.2
        
        if "prefer_afternoon" in meeting.constraints and meeting.constraints["prefer_afternoon"]:
            if 12 <= start_hour < 17:
                score += 0.2
        
        # Check participant availability
        availability_conflicts = self._check_participant_availability(meeting.participants, time_slot)
        if availability_conflicts == 0:
            score += 0.3
        else:
            score -= 0.1 * availability_conflicts
        
        # Check duration fit
        if time_slot.end_time - time_slot.start_time >= timedelta(minutes=meeting.duration):
            score += 0.2
        else:
            score -= 0.3
        
        return max(-1.0, min(1.0, score))
    
    def _check_participant_availability(self, participants: List[str], time_slot: TimeSlot) -> int:
        """Check participant availability conflicts"""
        conflicts = 0
        day_name = time_slot.start_time.strftime("%A")
        time_range = f"{day_name} {time_slot.start_time.strftime('%H:%M')}-{time_slot.end_time.strftime('%H:%M')}"
        
        for participant in participants:
            if participant in self.participant_availability:
                busy_times = self.participant_availability[participant]
                if any(busy_time in time_range for busy_time in busy_times):
                    conflicts += 1
        
        return conflicts
    
    def _apply_scheduling_action(self, action: Action, meeting: Meeting):
        """Apply scheduling action to meeting"""
        if action.action_type == "schedule":
            time_slot_id = action.parameters.get("time_slot_id", "")
            time_slot = self._find_time_slot(time_slot_id)
            
            if time_slot and not time_slot.occupied:
                meeting.scheduled_time = time_slot.start_time
                meeting.status = "scheduled"
                time_slot.occupied = True
                time_slot.meeting = meeting
        
        elif action.action_type == "skip":
            meeting.status = "skipped"
        
        elif action.action_type == "reschedule":
            meeting.status = "rescheduled"
        
        elif action.action_type == "cancel":
            meeting.status = "cancelled"
            if meeting.scheduled_time:
                # Free up the time slot
                for slot in self.time_slots:
                    if slot.meeting and slot.meeting.meeting_id == meeting.meeting_id:
                        slot.occupied = False
                        slot.meeting = None
                        break
    
    def _check_schedule_conflicts(self):
        """Check for scheduling conflicts"""
        self.schedule_conflicts = []
        
        for i, slot1 in enumerate(self.time_slots):
            if slot1.occupied and slot1.meeting:
                for j, slot2 in enumerate(self.time_slots):
                    if i != j and slot2.occupied and slot2.meeting:
                        # Check for overlapping participants
                        participants1 = set(slot1.meeting.participants)
                        participants2 = set(slot2.meeting.participants)
                        
                        if participants1 & participants2:  # Overlapping participants
                            # Check time overlap
                            if (slot1.start_time < slot2.end_time and slot2.start_time < slot1.end_time):
                                self.schedule_conflicts.append({
                                    "meeting1": slot1.meeting.meeting_id,
                                    "meeting2": slot2.meeting.meeting_id,
                                    "conflicting_participants": list(participants1 & participants2)
                                })
    
    def _find_time_slot(self, time_slot_id: str) -> Optional[TimeSlot]:
        """Find time slot by ID"""
        for slot in self.time_slots:
            slot_id = f"{slot.start_time.strftime('%Y-%m-%d_%H:%M')}"
            if slot_id == time_slot_id:
                return slot
        return None
    
    def _get_scheduled_meetings(self) -> List[Dict[str, Any]]:
        """Get list of scheduled meetings"""
        scheduled = []
        for meeting in self.meetings:
            if meeting.status == "scheduled":
                scheduled.append({
                    "meeting_id": meeting.meeting_id,
                    "title": meeting.title,
                    "scheduled_time": meeting.scheduled_time.isoformat() if meeting.scheduled_time else None,
                    "priority": meeting.priority
                })
        return scheduled
    
    def _get_available_time_slots(self) -> List[Dict[str, Any]]:
        """Get list of available time slots"""
        available = []
        for slot in self.time_slots:
            if not slot.occupied:
                available.append({
                    "slot_id": f"{slot.start_time.strftime('%Y-%m-%d_%H:%M')}",
                    "start_time": slot.start_time.isoformat(),
                    "end_time": slot.end_time.isoformat(),
                    "duration_minutes": int((slot.end_time - slot.start_time).total_seconds() / 60)
                })
        return available
    
    def _calculate_schedule_score(self) -> float:
        """Calculate overall schedule quality score"""
        if not self.meetings:
            return 0.0
        
        total_meetings = len(self.meetings)
        scheduled_meetings = len([m for m in self.meetings if m.status == "scheduled"])
        high_priority_scheduled = len([m for m in self.meetings if m.status == "scheduled" and m.priority == "high"])
        
        # Base score for scheduling
        schedule_rate = scheduled_meetings / total_meetings
        
        # Priority weighting
        high_priority_bonus = (high_priority_scheduled / max(1, len([m for m in self.meetings if m.priority == "high"]))) * 0.2
        
        # Conflict penalty
        conflict_penalty = min(0.3, len(self.schedule_conflicts) * 0.1)
        
        final_score = schedule_rate + high_priority_bonus - conflict_penalty
        
        return max(0.0, min(1.0, final_score))
    
    def _get_current_observation(self) -> Observation:
        """Get current observation"""
        current_meeting = None
        if self.current_meeting_index < len(self.meetings):
            current_meeting = self.meetings[self.current_meeting_index].__dict__
        
        return Observation(
            task_state={
                "current_meeting": current_meeting,
                "meetings_remaining": len(self.meetings) - self.current_meeting_index - 1,
                "scheduled_meetings": self._get_scheduled_meetings(),
                "time_slots_available": self._get_available_time_slots(),
                "participant_availability": self.participant_availability,
                "schedule_conflicts": self.schedule_conflicts[-3:] if self.schedule_conflicts else [],
                "total_conflicts": len(self.schedule_conflicts)
            },
            available_actions=self.get_available_actions(),
            step_count=self.current_step,
            max_steps=self.max_steps,
            metadata={"task": "scheduling", "difficulty": "hard"}
        )


class SchedulingGrader(TaskGrader):
    """Grader for scheduling task"""
    
    def grade(self, final_state: Dict[str, Any]) -> float:
        """Grade scheduling performance"""
        meetings = final_state.get("meetings", [])
        schedule_conflicts = final_state.get("schedule_conflicts", [])
        
        if not meetings:
            return 0.0
        
        # Convert to Meeting objects for easier processing
        meeting_objects = []
        for meeting_data in meetings:
            meeting = Meeting(**meeting_data)
            meeting_objects.append(meeting)
        
        total_meetings = len(meeting_objects)
        scheduled_meetings = len([m for m in meeting_objects if m.status == "scheduled"])
        high_priority_scheduled = len([m for m in meeting_objects if m.status == "scheduled" and m.priority == "high"])
        medium_priority_scheduled = len([m for m in meeting_objects if m.status == "scheduled" and m.priority == "medium"])
        
        # Calculate scores
        schedule_rate = scheduled_meetings / total_meetings
        high_priority_rate = high_priority_scheduled / max(1, len([m for m in meeting_objects if m.priority == "high"]))
        medium_priority_rate = medium_priority_scheduled / max(1, len([m for m in meeting_objects if m.priority == "medium"]))
        
        # Conflict penalty
        conflict_penalty = min(0.3, len(schedule_conflicts) * 0.05)
        
        # Weighted score
        final_score = (
            schedule_rate * 0.4 +
            high_priority_rate * 0.3 +
            medium_priority_rate * 0.2 +
            (1.0 - conflict_penalty) * 0.1
        )
        
        return max(0.0, min(1.0, final_score))
    
    def get_grading_criteria(self) -> Dict[str, Any]:
        """Return grading criteria"""
        return {
            "schedule_rate_weight": 0.4,
            "high_priority_weight": 0.3,
            "medium_priority_weight": 0.2,
            "conflict_penalty_weight": 0.1,
            "scoring": "Weighted combination of scheduling rate, priority handling, and conflict avoidance",
            "priorities": ["high", "medium", "low"],
            "target_score": 0.8
        }
