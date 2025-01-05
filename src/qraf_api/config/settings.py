from pydantic_settings import BaseSettings
from typing import List, Optional
from functools import lru_cache

class Settings(BaseSettings):
    """API service configuration settings"""
    
    # API Settings
    API_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    DEBUG: bool = False
    
    # CORS Settings
    ALLOWED_ORIGINS: List[str] = ["*"]
    ALLOWED_METHODS: List[str] = ["*"]
    ALLOWED_HEADERS: List[str] = ["*"]
    
    # Quantum Settings
    HIDDEN_SIZE: int = 768
    NUM_ATTENTION_HEADS: int = 12
    COMPRESSION_RATE: float = 0.8
    COHERENCE_THRESHOLD: float = 0.7
    PHASE_PRESERVATION: bool = True
    
    # CUDA Settings
    USE_CUDA: bool = True
    CUDA_VISIBLE_DEVICES: Optional[str] = None
    
    # Security Settings
    API_KEY_HEADER: str = "X-API-Key"
    JWT_SECRET_KEY: str = "your-secret-key-here"  # Change in production
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Rate Limiting
    RATE_LIMIT_CALLS: int = 100
    RATE_LIMIT_PERIOD: int = 60  # seconds
    
    # Memory Management
    MAX_STATES_PER_USER: int = 1000
    STATE_TTL_SECONDS: int = 3600  # 1 hour
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

# Example usage:
# settings = get_settings()
# print(settings.API_VERSION) 