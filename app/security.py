import hashlib
import secrets


API_KEY_PREFIX = "sk-"


def generate_api_key() -> str:
    """
    Generate a cryptographically secure API key.
    """
    random_part = secrets.token_urlsafe(32)

    return f"{API_KEY_PREFIX}{random_part}"


def hash_api_key(api_key: str) -> str:
    """
    SHA-256 hash of the API key.
    """
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def get_key_prefix(api_key: str) -> str:
    """
    Store only a short prefix for identification.
    """
    return api_key[:12]
