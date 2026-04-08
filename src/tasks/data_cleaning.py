import random
import pandas as pd
from typing import Dict, List, Any, Tuple
from ..core import BaseEnvironment, Observation, Action, Reward, Info, TaskGrader


class DataCleaningEnvironment(BaseEnvironment):
    """Data cleaning environment - medium difficulty"""
    
    def __init__(self, max_steps: int = 30):
        super().__init__(max_steps)
        self.task_name = "data_cleaning"
        self.difficulty = "medium"
        self.original_data = None
        self.current_data = None
        self.cleaning_history = []
        self.target_columns = ["name", "email", "phone", "age", "city"]
        
    def reset(self) -> Observation:
        """Reset environment with messy dataset"""
        self.current_step = 0
        self.original_data = self._generate_messy_dataset()
        self.current_data = self.original_data.copy()
        self.cleaning_history = []
        
        return Observation(
            task_state={
                "data_preview": self._get_data_preview(),
                "data_quality": self._assess_data_quality(),
                "column_info": self._get_column_info(),
                "cleaning_operations": len(self.cleaning_history)
            },
            available_actions=self.get_available_actions(),
            step_count=self.current_step,
            max_steps=self.max_steps,
            metadata={"task": "data_cleaning", "difficulty": "medium"}
        )
    
    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Info]:
        """Process data cleaning action"""
        self.current_step += 1
        
        if not self.validate_action(action):
            return self._get_current_observation(), Reward(value=-0.1, reason="Invalid action"), False, \
                   Info(errors=["Invalid action"])
        
        # Apply cleaning operation
        previous_quality = self._assess_data_quality()
        operation_result = self._apply_cleaning_operation(action)
        self.cleaning_history.append(operation_result)
        
        # Calculate reward based on improvement
        current_quality = self._assess_data_quality()
        quality_improvement = current_quality["overall_score"] - previous_quality["overall_score"]
        
        reward_value = max(-0.2, min(0.5, quality_improvement * 2))
        reward_reason = f"Quality change: {quality_improvement:+.3f}"
        
        # Check if task is complete (high quality achieved)
        done = current_quality["overall_score"] >= 0.9 or self.current_step >= self.max_steps
        
        observation = self._get_current_observation()
        info = Info(
            task_complete=done,
            score=current_quality["overall_score"],
            metadata={
                "quality_improvement": quality_improvement,
                "operations_count": len(self.cleaning_history)
            }
        )
        
        return observation, Reward(value=reward_value, reason=reward_reason), done, info
    
    def state(self) -> Dict[str, Any]:
        """Return current environment state"""
        return {
            "original_data": self.original_data.to_dict() if self.original_data is not None else None,
            "current_data": self.current_data.to_dict() if self.current_data is not None else None,
            "cleaning_history": self.cleaning_history,
            "step_count": self.current_step,
            "max_steps": self.max_steps
        }
    
    def validate_action(self, action: Action) -> bool:
        """Validate data cleaning action"""
        valid_actions = {"remove_duplicates", "fill_missing", "standardize_text", "validate_format", "remove_outliers"}
        return action.action_type in valid_actions
    
    def get_available_actions(self) -> List[str]:
        """Get available cleaning actions"""
        return ["remove_duplicates", "fill_missing", "standardize_text", "validate_format", "remove_outliers"]
    
    def _generate_messy_dataset(self) -> pd.DataFrame:
        """Generate a messy dataset for cleaning"""
        data = {
            "name": [
                "John Doe", "jane smith", "   Alice Johnson   ", "Bob", "Eve Wilson",
                "Charlie Brown", "Diana", "Frank Miller", "Grace Lee", "Henry",
                "John Doe", "  alice johnson  ", "Ivy Chen", "Jack", "Karen White"
            ],
            "email": [
                "john@email.com", "jane.smith@company.com", "alice@work.org", "invalid-email", "eve@domain.com",
                "charlie@company.com", "diana@", "frank@work.org", "grace.lee@company.com", "henry@work.org",
                "john@email.com", "alice.johnson@work.org", "ivy@company.com", "jack@", "karen.white@domain.com"
            ],
            "phone": [
                "555-1234", "555-5678", "555-9012", "123", "555-3456",
                "(555) 7890", "555-2345", "phone", "555-6789", "555-0123",
                "555-1234", "5559012", "555-4567", "555-8901", "555-2345"
            ],
            "age": [
                25, 30, 35, 150, 28,
                45, -5, 32, 29, 40,
                25, 35, 27, 33, 45
            ],
            "city": [
                "New York", "los angeles", "Chicago", "NYC", "Houston",
                "Phoenix", "Boston", "ATL", "Seattle", "Miami",
                "New York", "chicago", "Denver", "Portland", "Boston"
            ]
        }
        
        df = pd.DataFrame(data)
        
        # Add some missing values
        for col in ["name", "email", "phone"]:
            missing_indices = random.sample(range(len(df)), 2)
            df.loc[missing_indices, col] = None
        
        return df
    
    def _apply_cleaning_operation(self, action: Action) -> Dict[str, Any]:
        """Apply a cleaning operation to the data"""
        operation = action.action_type
        parameters = action.parameters
        
        result = {"operation": operation, "parameters": parameters, "rows_affected": 0}
        
        if operation == "remove_duplicates":
            before_count = len(self.current_data)
            self.current_data = self.current_data.drop_duplicates()
            result["rows_affected"] = before_count - len(self.current_data)
            
        elif operation == "fill_missing":
            column = parameters.get("column", "")
            method = parameters.get("method", "mean")
            
            if column in self.current_data.columns:
                if self.current_data[column].dtype in ['int64', 'float64']:
                    if method == "mean":
                        fill_value = self.current_data[column].mean()
                    elif method == "median":
                        fill_value = self.current_data[column].median()
                    else:
                        fill_value = 0
                else:
                    fill_value = "Unknown"
                
                missing_count = self.current_data[column].isna().sum()
                self.current_data[column].fillna(fill_value, inplace=True)
                result["rows_affected"] = missing_count
                
        elif operation == "standardize_text":
            column = parameters.get("column", "")
            
            if column in self.current_data.columns:
                # Remove leading/trailing whitespace and capitalize properly
                self.current_data[column] = self.current_data[column].str.strip().str.title()
                result["rows_affected"] = len(self.current_data)
                
        elif operation == "validate_format":
            column = parameters.get("column", "")
            format_type = parameters.get("format", "")
            
            if column in self.current_data.columns:
                if format_type == "email" and column == "email":
                    # Basic email validation
                    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                    invalid_mask = ~self.current_data[column].str.match(pattern, na=False)
                    invalid_count = invalid_mask.sum()
                    self.current_data.loc[invalid_mask, column] = None
                    result["rows_affected"] = invalid_count
                    
                elif format_type == "phone" and column == "phone":
                    # Standardize phone format
                    self.current_data[column] = self.current_data[column].str.replace(r'[^\d]', '', regex=True)
                    # Keep only valid phone numbers (10 digits)
                    invalid_mask = (self.current_data[column].str.len() != 10)
                    invalid_count = invalid_mask.sum()
                    self.current_data.loc[invalid_mask, column] = None
                    result["rows_affected"] = invalid_count
                    
        elif operation == "remove_outliers":
            column = parameters.get("column", "")
            
            if column in self.current_data.columns and self.current_data[column].dtype in ['int64', 'float64']:
                Q1 = self.current_data[column].quantile(0.25)
                Q3 = self.current_data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (self.current_data[column] < lower_bound) | (self.current_data[column] > upper_bound)
                outlier_count = outlier_mask.sum()
                self.current_data.loc[outlier_mask, column] = None
                result["rows_affected"] = outlier_count
        
        return result
    
    def _assess_data_quality(self) -> Dict[str, Any]:
        """Assess current data quality"""
        if self.current_data is None:
            return {"overall_score": 0.0}
        
        total_cells = len(self.current_data) * len(self.current_data.columns)
        missing_cells = self.current_data.isna().sum().sum()
        duplicate_rows = self.current_data.duplicated().sum()
        
        # Quality metrics
        completeness = 1.0 - (missing_cells / total_cells)
        uniqueness = 1.0 - (duplicate_rows / len(self.current_data))
        
        # Format validation
        format_score = 0.0
        if "email" in self.current_data.columns:
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            valid_emails = self.current_data["email"].str.match(email_pattern, na=False).sum()
            total_emails = self.current_data["email"].notna().sum()
            if total_emails > 0:
                format_score += valid_emails / total_emails
        
        if "phone" in self.current_data.columns:
            valid_phones = self.current_data["phone"].str.match(r'^\d{10}$', na=False).sum()
            total_phones = self.current_data["phone"].notna().sum()
            if total_phones > 0:
                format_score += valid_phones / total_phones
        
        format_score = format_score / 2  # Average of email and phone scores
        
        # Overall score
        overall_score = (completeness * 0.4 + uniqueness * 0.3 + format_score * 0.3)
        
        return {
            "overall_score": overall_score,
            "completeness": completeness,
            "uniqueness": uniqueness,
            "format_score": format_score,
            "missing_cells": missing_cells,
            "duplicate_rows": duplicate_rows
        }
    
    def _get_data_preview(self) -> Dict[str, Any]:
        """Get preview of current data"""
        if self.current_data is None:
            return {}
        
        return {
            "shape": self.current_data.shape,
            "columns": list(self.current_data.columns),
            "sample_rows": self.current_data.head(3).to_dict("records"),
            "data_types": self.current_data.dtypes.to_dict()
        }
    
    def _get_column_info(self) -> Dict[str, Any]:
        """Get information about each column"""
        if self.current_data is None:
            return {}
        
        column_info = {}
        for col in self.current_data.columns:
            column_info[col] = {
                "type": str(self.current_data[col].dtype),
                "missing_count": self.current_data[col].isna().sum(),
                "unique_count": self.current_data[col].nunique(),
                "sample_values": self.current_data[col].dropna().head(3).tolist()
            }
        
        return column_info
    
    def _get_current_observation(self) -> Observation:
        """Get current observation"""
        return Observation(
            task_state={
                "data_preview": self._get_data_preview(),
                "data_quality": self._assess_data_quality(),
                "column_info": self._get_column_info(),
                "cleaning_operations": len(self.cleaning_history),
                "recent_operations": self.cleaning_history[-3:] if self.cleaning_history else []
            },
            available_actions=self.get_available_actions(),
            step_count=self.current_step,
            max_steps=self.max_steps,
            metadata={"task": "data_cleaning", "difficulty": "medium"}
        )


class DataCleaningGrader(TaskGrader):
    """Grader for data cleaning task"""
    
    def grade(self, final_state: Dict[str, Any]) -> float:
        """Grade data cleaning performance"""
        current_data = final_state.get("current_data")
        
        if current_data is None:
            return 0.0
        
        df = pd.DataFrame(current_data)
        
        # Calculate final quality metrics
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isna().sum().sum()
        duplicate_rows = df.duplicated().sum()
        
        completeness = 1.0 - (missing_cells / total_cells)
        uniqueness = 1.0 - (duplicate_rows / len(df))
        
        # Format validation
        format_score = 0.0
        if "email" in df.columns:
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            valid_emails = df["email"].str.match(email_pattern, na=False).sum()
            total_emails = df["email"].notna().sum()
            if total_emails > 0:
                format_score += valid_emails / total_emails
        
        if "phone" in df.columns:
            valid_phones = df["phone"].str.match(r'^\d{10}$', na=False).sum()
            total_phones = df["phone"].notna().sum()
            if total_phones > 0:
                format_score += valid_phones / total_phones
        
        format_score = format_score / 2 if format_score > 0 else 0
        
        # Overall score
        final_score = (completeness * 0.4 + uniqueness * 0.3 + format_score * 0.3)
        
        return min(1.0, max(0.0, final_score))
    
    def get_grading_criteria(self) -> Dict[str, Any]:
        """Return grading criteria"""
        return {
            "completeness_weight": 0.4,
            "uniqueness_weight": 0.3,
            "format_weight": 0.3,
            "target_quality": 0.9,
            "scoring": "Weighted average of data completeness, uniqueness, and format validation"
        }
