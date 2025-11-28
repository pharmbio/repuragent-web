import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Keys
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Memory directory setup
MEMORY_DIR = Path("backend/memory")
MEMORY_DIR.mkdir(exist_ok=True)
DEMO_THREADS_FILE = MEMORY_DIR / "demo_threads.json"

# Database configuration
# PostgreSQL for main short-term memory checkpointer
DATABASE_URL = os.environ.get("DATABASE_URL")

# Authentication & security
AUTH_JWT_SECRET = os.environ.get("AUTH_JWT_SECRET", "changeme-secret")
AUTH_JWT_EXPIRES_MINUTES = int(os.environ.get("AUTH_JWT_EXPIRES_MINUTES", "60"))
AUTH_REFRESH_EXPIRES_DAYS = int(os.environ.get("AUTH_REFRESH_EXPIRES_DAYS", "7"))
AUTH_SESSION_COOKIE = os.environ.get("AUTH_SESSION_COOKIE", "repuragent_session")
AUTH_SESSION_COOKIE_SECURE = os.environ.get("AUTH_SESSION_COOKIE_SECURE", "false").lower() == "true"
AUTH_PEPPER = os.environ.get("AUTH_PEPPER", "")
EMAIL_SENDER = os.environ.get("EMAIL_SENDER", "no-reply@repuragent.local")
EMAIL_BASE_URL = os.environ.get("EMAIL_BASE_URL", "http://localhost:7860")
EMAIL_PROVIDER_API_KEY = os.environ.get("EMAIL_PROVIDER_API_KEY")
EMAIL_PROVIDER_API_URL = os.environ.get("EMAIL_PROVIDER_API_URL")
EMAIL_PROVIDER_USERNAME = os.environ.get("EMAIL_PROVIDER_USERNAME")
VERIFICATION_TOKEN_TTL_HOURS = int(os.environ.get("VERIFICATION_TOKEN_TTL_HOURS", "24"))
RESET_TOKEN_TTL_HOURS = int(os.environ.get("RESET_TOKEN_TTL_HOURS", "1"))
RESULT_RETENTION_DAYS = int(os.environ.get("RESULT_RETENTION_DAYS", "3"))
FILE_DOWNLOAD_SECRET = os.environ.get("FILE_DOWNLOAD_SECRET", AUTH_JWT_SECRET)
FILE_DOWNLOAD_TOKEN_TTL_SECONDS = int(os.environ.get("FILE_DOWNLOAD_TOKEN_TTL_SECONDS", "600"))

UI_QUEUE_MAX_SIZE = 128
UI_CONCURRENCY_LIMIT = 8

# Application settings
APP_TITLE = "Repuragent"
LOGO_PATH = "images/logo.png"
RECURSION_LIMIT = 100
