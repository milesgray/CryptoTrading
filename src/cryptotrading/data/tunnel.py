"""
SSH Tunnel utilities for connecting to remote PostgreSQL/TimescaleDB instances.
Enables transparent, secure access from local Jupyter notebooks and Python scripts
to databases running inside Docker Compose stacks on remote cloud VMs.
"""

import os
import contextlib
from typing import Generator, Tuple, Optional, Any, Dict
# Compatibility shim for paramiko >= 3.0 / 4.x / 5.x where DSSKey was removed
import paramiko
if not hasattr(paramiko, "DSSKey"):
    try:
        from paramiko.dsskey import DSSKey
        paramiko.DSSKey = DSSKey
    except Exception:
        class DummyDSSKey:
            pass
        paramiko.DSSKey = DummyDSSKey

from sshtunnel import SSHTunnelForwarder
from sqlalchemy import create_engine, Engine
import asyncpg

# Default connection settings
DEFAULT_SSH_HOST = os.getenv("REMOTE_SSH_HOST", "50.117.53.113")
DEFAULT_SSH_USER = os.getenv("REMOTE_SSH_USER", "cloud")
DEFAULT_SSH_PORT = int(os.getenv("REMOTE_SSH_PORT", "22"))
DEFAULT_SSH_PKEY = os.path.expanduser(os.getenv("REMOTE_SSH_KEY", "~/.ssh/id_rsa"))

DEFAULT_DB_HOST = "127.0.0.1"
DEFAULT_DB_PORT = 5432
DEFAULT_DB_NAME = os.getenv("POSTGRES_DB", "crypto_trading")
DEFAULT_DB_USER = os.getenv("POSTGRES_USER", "postgres")
DEFAULT_DB_PASSWORD = os.getenv("POSTGRES_REMOTE_PASSWORD", "postgres")


def open_tunnel(
    ssh_host: str = DEFAULT_SSH_HOST,
    ssh_user: str = DEFAULT_SSH_USER,
    ssh_port: int = DEFAULT_SSH_PORT,
    ssh_pkey: Optional[str] = None,
    remote_bind_host: str = DEFAULT_DB_HOST,
    remote_bind_port: int = DEFAULT_DB_PORT,
    local_bind_port: Optional[int] = None,
) -> SSHTunnelForwarder:
    """
    Open and start an SSH tunnel to the remote host.
    
    Args:
        ssh_host: Hostname or IP of the cloud VM.
        ssh_user: SSH username on the cloud VM.
        ssh_port: SSH port (default 22).
        ssh_pkey: Path to private key file (if None, attempts ~/.ssh/id_rsa or ssh-agent).
        remote_bind_host: Target host from the VM's perspective (usually 127.0.0.1).
        remote_bind_port: Target port on the remote host (Postgres default 5432).
        local_bind_port: Local port to bind to. If None, binds to an available random port.
        
    Returns:
        An active SSHTunnelForwarder instance.
    """
    pkey_path = ssh_pkey or (DEFAULT_SSH_PKEY if os.path.exists(DEFAULT_SSH_PKEY) else None)
    
    tunnel_kwargs: Dict[str, Any] = {
        "ssh_address_or_host": (ssh_host, ssh_port),
        "ssh_username": ssh_user,
        "remote_bind_address": (remote_bind_host, remote_bind_port),
    }
    if pkey_path:
        tunnel_kwargs["ssh_pkey"] = pkey_path
    if local_bind_port:
        tunnel_kwargs["local_bind_address"] = ("127.0.0.1", local_bind_port)
        
    tunnel = SSHTunnelForwarder(**tunnel_kwargs)
    tunnel.start()
    return tunnel


def get_remote_engine(
    ssh_host: str = DEFAULT_SSH_HOST,
    ssh_user: str = DEFAULT_SSH_USER,
    db_name: str = DEFAULT_DB_NAME,
    db_user: str = DEFAULT_DB_USER,
    db_password: str = DEFAULT_DB_PASSWORD,
    ssh_pkey: Optional[str] = None,
    echo: bool = False,
) -> Tuple[Engine, SSHTunnelForwarder]:
    """
    Establish an SSH tunnel and create a SQLAlchemy Engine.
    
    Returns:
        tuple: (sqlalchemy.Engine, SSHTunnelForwarder)
        Remember to call tunnel.stop() when finished or use remote_db_tunnel() context manager.
    """
    tunnel = open_tunnel(
        ssh_host=ssh_host,
        ssh_user=ssh_user,
        ssh_pkey=ssh_pkey,
    )
    conn_str = f"postgresql+psycopg2://{db_user}:{db_password}@127.0.0.1:{tunnel.local_bind_port}/{db_name}"
    engine = create_engine(conn_str, echo=echo)
    return engine, tunnel


@contextlib.contextmanager
def remote_db_tunnel(
    ssh_host: str = DEFAULT_SSH_HOST,
    ssh_user: str = DEFAULT_SSH_USER,
    db_name: str = DEFAULT_DB_NAME,
    db_user: str = DEFAULT_DB_USER,
    db_password: str = DEFAULT_DB_PASSWORD,
    ssh_pkey: Optional[str] = None,
    echo: bool = False,
) -> Generator[Tuple[Engine, SSHTunnelForwarder], None, None]:
    """
    Context manager that starts an SSH tunnel, yields (engine, tunnel), and stops the tunnel on exit.
    
    Example:
        with remote_db_tunnel() as (engine, tunnel):
            df = pd.read_sql("SELECT * FROM crypto_assets", engine)
    """
    engine, tunnel = get_remote_engine(
        ssh_host=ssh_host,
        ssh_user=ssh_user,
        db_name=db_name,
        db_user=db_user,
        db_password=db_password,
        ssh_pkey=ssh_pkey,
        echo=echo,
    )
    try:
        yield engine, tunnel
    finally:
        engine.dispose()
        tunnel.stop()


async def get_asyncpg_connection(
    tunnel: SSHTunnelForwarder,
    db_name: str = DEFAULT_DB_NAME,
    db_user: str = DEFAULT_DB_USER,
    db_password: str = DEFAULT_DB_PASSWORD,
) -> asyncpg.Connection:
    """
    Create an asyncpg connection over an active SSH tunnel.
    """
    return await asyncpg.connect(
        host="127.0.0.1",
        port=tunnel.local_bind_port,
        user=db_user,
        password=db_password,
        database=db_name,
    )


def test_connection() -> bool:
    """
    Quick test to verify SSH tunnel and database query functionality.
    """
    try:
        with remote_db_tunnel() as (engine, tunnel):
            with engine.connect() as conn:
                from sqlalchemy import text
                result = conn.execute(text("SELECT version(), current_database();")).fetchone()
                print(f"Connected successfully via SSH Tunnel (Local port {tunnel.local_bind_port})!")
                print(f"Database Info: {result}")
                return True
    except Exception as e:
        print(f"Connection test failed: {e}")
        return False


if __name__ == "__main__":
    test_connection()
