import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class DatabaseSettings:
    """Supabase Database & Auth Settings for StudySnap AI."""
    
    # Supabase Connection Parameters
    SUPABASE_URL: str = (
        os.getenv("SUPABASE_URL") or 
        os.getenv("VITE_SUPABASE_URL") or 
        ""
    )
    
    SUPABASE_PUBLISHABLE_KEY: str = (
        os.getenv("SUPABASE_PUBLISHABLE_KEY") or 
        os.getenv("SUPABASE_ANON_KEY") or 
        os.getenv("VITE_SUPABASE_PUBLISHABLE_KEY") or 
        os.getenv("VITE_SUPABASE_ANON_KEY") or 
        ""
    )
    
    SUPABASE_SECRET_KEY: str = (
        os.getenv("SUPABASE_SECRET_KEY") or 
        os.getenv("SUPABASE_SERVICE_ROLE_KEY") or 
        ""
    )
    
    SUPABASE_JWT_SECRET: str = (
        os.getenv("SUPABASE_JWT_SECRET") or 
        os.getenv("JWT_SECRET") or 
        ""
    )

    # Supabase Storage Bucket for study assets
    SUPABASE_STORAGE_BUCKET: str = os.getenv("SUPABASE_STORAGE_BUCKET", "studysnap-assets")

    @property
    def is_configured(self) -> bool:
        """Verify whether valid Supabase credentials have been configured."""
        return bool(
            self.SUPABASE_URL 
            and self.SUPABASE_URL.startswith("http") 
            and "your-project-id" not in self.SUPABASE_URL
            and (self.SUPABASE_PUBLISHABLE_KEY or self.SUPABASE_SECRET_KEY)
        )

settings = DatabaseSettings()
