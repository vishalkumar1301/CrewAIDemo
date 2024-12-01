from instructor import OpenAISchema
from pydantic import BaseModel, Field
from typing import List, Dict

class Topic(OpenAISchema):
    """A topic with its verification points"""
    name: str = Field(..., description="Name of the topic")
    exact_text_mention: str = Field(..., description="Exact text mention of the topic in the text")
    description: str = Field(..., description="Brief description of the factual claim")
    verification_points: List[str] = Field(..., description="List of specific questions or details to verify")

class AnalyzeTextOutput(OpenAISchema):
    """Output for analyze_text_task"""
    points: List[str] = Field(..., description="List of main points extracted from the text")

class TopicInfo(BaseModel):
    description: str = Field(..., description="Brief description of the factual claim")
    verification_points: List[str] = Field(..., description="List of specific questions or details to verify")

class VerifyCredibilityOutput(OpenAISchema):
    """Output for verify_credibility_task"""
    topics: Dict[str, TopicInfo] = Field(..., description="Dictionary of topics and their verification points")

class SelectObjectiveOutput(OpenAISchema):
    """Output for select_objective_task"""
    topics: Dict[str, TopicInfo] = Field(..., description="Dictionary of objective topics and their verification points")
