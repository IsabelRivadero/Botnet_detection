from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class UserPayload(BaseModel):
    user_id: int
    screen_name: str
    created_at: datetime


class TweetPayload(BaseModel):
    tweet_id: int
    user_id: int
    created_at: datetime
    text: str
    mentions: Optional[List[str]] = Field(default_factory=list)


class FitRequest(BaseModel):
    data_dir: str = "data"
    as_of: datetime
    lookback_days: int = 30
    alpha: float = 0.05
    model_path: str = "model.joblib"


class ScoreRequest(BaseModel):
    data_dir: str = "data"
    as_of: datetime
    lookback_days: int = 30
    model_path: str = "model.joblib"
    out_path: str = "flagged_communities.csv"


class AnalyzeRequest(BaseModel):
    users: List[UserPayload]
    tweets: List[TweetPayload]
    as_of: datetime
    lookback_days: int = 30
    model_path: str = "model.joblib"


class CommunityDetail(BaseModel):
    community_id: int
    size: int
    quadrant: str
    is_botnet: bool
    scores: Dict[str, float]
    members: List[int]


class AnalysisResponse(BaseModel):
    status: str
    total_communities: int
    flagged_communities: int
    communities: List[CommunityDetail]


class HealthResponse(BaseModel):
    status: str
    version: str