"""
Credentials management for WebSocket Paper Tester.
Loads API keys from environment files for exchange access.
"""

import os
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class KrakenCredentials:
    """Kraken API credentials."""
    api_key: str
    private_key: str

    @property
    def is_valid(self) -> bool:
        """Check if credentials are present."""
        return bool(self.api_key and self.private_key)


def load_env_file(env_path: Path) -> Dict[str, str]:
    """
    Load environment variables from a .env file.

    Args:
        env_path: Path to .env file

    Returns:
        Dictionary of key-value pairs
    """
    env_vars = {}

    if not env_path.exists():
        return env_vars

    try:
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                # Parse key=value
                if '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
    except Exception as e:
        print(f"[Credentials] Error loading {env_path}: {e}")

    return env_vars


def load_kraken_credentials(
    env_file: str = None,
    config_dir: str = None
) -> Optional[KrakenCredentials]:
    """
    Load Kraken API credentials from environment or .env file.

    Priority:
    1. Environment variables (KRAKEN_API_KEY, KRAKEN_PRIVATE_KEY)
    2. .env file in specified config_dir
    3. .env file in project's config/ directory
    4. .env file in parent's config/ directory

    Args:
        env_file: Explicit path to .env file
        config_dir: Directory containing .env file

    Returns:
        KrakenCredentials if found, None otherwise
    """
    api_key = os.environ.get('KRAKEN_API_KEY', '')
    private_key = os.environ.get('KRAKEN_PRIVATE_KEY', '')

    # If not in environment, try .env file
    if not (api_key and private_key):
        env_paths = []

        if env_file:
            env_paths.append(Path(env_file))
        if config_dir:
            env_paths.append(Path(config_dir) / '.env')

        # Default search paths
        project_root = Path(__file__).parent.parent
        env_paths.extend([
            project_root / 'config' / '.env',
            project_root / '.env',
            project_root.parent / 'config' / '.env',  # Parent project config
        ])

        for env_path in env_paths:
            if env_path.exists():
                env_vars = load_env_file(env_path)
                api_key = api_key or env_vars.get('KRAKEN_API_KEY', '')
                private_key = private_key or env_vars.get('KRAKEN_PRIVATE_KEY', '')
                if api_key and private_key:
                    print(f"[Credentials] Loaded Kraken credentials from {env_path}")
                    break

    if not (api_key and private_key):
        print("[Credentials] Warning: Kraken credentials not found")
        return None

    return KrakenCredentials(api_key=api_key, private_key=private_key)


def get_kraken_ws_token(credentials: KrakenCredentials) -> Optional[str]:
    """
    Get WebSocket authentication token from Kraken REST API.

    This token is needed for private WebSocket channels (own trades, orders).

    Args:
        credentials: Kraken API credentials

    Returns:
        WebSocket token string, or None if failed
    """
    import hmac
    import hashlib
    import base64
    import time
    import urllib.parse

    try:
        import requests
    except ImportError:
        print("[Credentials] requests library not installed, cannot get WS token")
        return None

    if not credentials or not credentials.is_valid:
        return None

    url = "https://api.kraken.com/0/private/GetWebSocketsToken"
    nonce = str(int(time.time() * 1000))

    post_data = {
        'nonce': nonce
    }

    # Create signature
    post_data_str = urllib.parse.urlencode(post_data)
    encoded = (nonce + post_data_str).encode()
    message = '/0/private/GetWebSocketsToken'.encode() + hashlib.sha256(encoded).digest()

    try:
        secret = base64.b64decode(credentials.private_key)
        signature = hmac.new(secret, message, hashlib.sha512)
        sig_digest = base64.b64encode(signature.digest()).decode()
    except Exception as e:
        print(f"[Credentials] Signature generation failed: {e}")
        return None

    headers = {
        'API-Key': credentials.api_key,
        'API-Sign': sig_digest,
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    try:
        response = requests.post(url, headers=headers, data=post_data, timeout=10)
        result = response.json()

        if result.get('error'):
            print(f"[Credentials] Kraken API error: {result['error']}")
            return None

        token = result.get('result', {}).get('token')
        if token:
            print("[Credentials] Successfully obtained Kraken WebSocket token")
            return token
        else:
            print("[Credentials] No token in response")
            return None

    except requests.RequestException as e:
        print(f"[Credentials] Request failed: {e}")
        return None
    except Exception as e:
        print(f"[Credentials] Unexpected error: {e}")
        return None


def verify_kraken_credentials(credentials: KrakenCredentials) -> bool:
    """
    Verify Kraken credentials by making a test API call.

    Args:
        credentials: Kraken API credentials

    Returns:
        True if credentials are valid
    """
    token = get_kraken_ws_token(credentials)
    return token is not None
