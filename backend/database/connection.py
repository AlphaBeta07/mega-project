import logging
from typing import Optional
from supabase import create_client, Client
from .config import settings

logger = logging.getLogger("studysnap.supabase")

class SupabaseManager:
    """Connection manager for Supabase PostgreSQL and Storage."""
    
    def __init__(self):
        self._anon_client: Optional[Client] = None
        self._service_client: Optional[Client] = None

    def get_client(self, access_token: Optional[str] = None) -> Optional[Client]:
        """
        Get or initialize a Supabase client.
        If access_token is provided, sets postgrest authentication to enforce RLS.
        """
        if not settings.is_configured:
            return None

        key = settings.SUPABASE_PUBLISHABLE_KEY or settings.SUPABASE_SECRET_KEY
        if not self._anon_client:
            try:
                self._anon_client = create_client(settings.SUPABASE_URL, key)
                logger.info(f"Initialized Supabase client for {settings.SUPABASE_URL}")
            except Exception as e:
                logger.error(f"Failed to initialize Supabase client: {e}")
                return None

        client = self._anon_client
        if access_token and client:
            try:
                client.postgrest.auth(access_token)
            except Exception as auth_err:
                logger.warning(f"Failed to set postgrest bearer token: {auth_err}")

        return client

    def get_service_client(self) -> Optional[Client]:
        """Get trusted service-role client for background administration."""
        if not settings.is_configured or not settings.SUPABASE_SECRET_KEY:
            return None

        if not self._service_client:
            try:
                self._service_client = create_client(
                    settings.SUPABASE_URL, 
                    settings.SUPABASE_SECRET_KEY
                )
                logger.info("Initialized Supabase service-role client.")
            except Exception as e:
                logger.error(f"Failed to initialize Supabase service client: {e}")
                return None

        return self._service_client

    async def ping(self) -> bool:
        """Check if Supabase instance is online and reachable."""
        if not settings.is_configured:
            return False
        try:
            client = self.get_client()
            if not client:
                return False
            # Simple query to verify connection
            res = client.table("notebooks").select("id").limit(1).execute()
            return True
        except Exception as e:
            logger.debug(f"Supabase ping note: {e}")
            # If error is table not found or auth error, Supabase itself is still reached
            err_msg = str(e).lower()
            if "pgrst" in err_msg or "unauthorized" in err_msg or "not found" in err_msg:
                return True
            return False

supabase_manager = SupabaseManager()

def get_supabase(access_token: Optional[str] = None) -> Optional[Client]:
    """Dependency helper to get active Supabase client."""
    return supabase_manager.get_client(access_token)

async def ping_db() -> bool:
    """Verify database connection status."""
    return await supabase_manager.ping()

class _CompatibilityDB:
    def __getitem__(self, _):
        return self
    def replace_one(self, *args, **kwargs):
        return None
    def find(self, *args, **kwargs):
        return []
    def delete_one(self, *args, **kwargs):
        return None

_compat_db = _CompatibilityDB()

def get_sync_db():
    """Compatibility stub for legacy MongoDB sync calls."""
    return _compat_db

