from pydantic import BaseModel, Field
from typing import Optional, List

class ClassifyRequest(BaseModel):
    text: str = Field(..., description="The raw content of the Telegram message")
    chat_title: Optional[str] = Field(None, description="Optional name of the chat or group")
    sender_name: Optional[str] = Field(None, description="Optional name/username of the sender")

class ClassifyResponse(BaseModel):
    valuable: bool = Field(..., description="Indicates if the message has high business/project value")
    category: str = Field(..., description="hiring | project_request | collaboration | irrelevant")
    score: float = Field(..., description="The final positive score minus any negative adjustments")
    confidence: float = Field(..., description="The confidence score of the classification, between 0.0 and 1.0")
    reasons: List[str] = Field(..., description="List of positive indicators and reasons for the decision")
    matched_keywords: List[str] = Field(..., description="List of positive keywords matched during classification")
    negative_reasons: List[str] = Field(..., description="List of negative reasons/penalties identified")
