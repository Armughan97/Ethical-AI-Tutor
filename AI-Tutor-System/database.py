"""
Database configuration and session management for the tutoring system.
"""

import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from typing import Generator
from contextlib import contextmanager

from models import Base

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DATABASE_DIR = os.path.join(os.path.dirname(__file__), "data")
DATABASE_URL = f"sqlite:///{DATABASE_DIR}/tutoring_metrics.db"

# Create data directory if it doesn't exist
os.makedirs(DATABASE_DIR, exist_ok=True)

# SQLAlchemy engine configuration
engine = create_engine(
    DATABASE_URL,
    connect_args={
        "check_same_thread": False,  # Allow SQLite to be used in multi-threaded environment
        "timeout": 20  # Set timeout for database operations
    },
    poolclass=StaticPool,  # Use static pool for SQLite
    echo=False  # Set to True for SQL query logging
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def create_tables() -> None:
    """
    Create all database tables if they don't exist.
    """
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise


def get_db_session() -> Generator[Session, None, None]:
    """
    Dependency function to get database session.
    Yields a database session and ensures it's closed after use.
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.
    Use this for direct database operations outside of FastAPI dependency injection.
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        logger.error(f"Database context error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def init_database() -> None:
    """
    Initialize the database by creating tables.
    Call this function when starting the application.
    """
    logger.info("Initializing database...")
    create_tables()
    logger.info("Database initialization completed")


def reset_database() -> None:
    """
    Reset the database by dropping and recreating all tables.
    Use with caution - this will delete all data!
    """
    logger.warning("Resetting database - all data will be lost!")
    Base.metadata.drop_all(bind=engine)
    create_tables()
    logger.info("Database reset completed")


def get_db_stats() -> dict:
    """
    Get basic database statistics.
    """
    with get_db_context() as db:
        from models import Interaction
        
        total_interactions = db.query(Interaction).count()
        unique_users = db.query(Interaction.user_id).distinct().count()
        unique_personas = db.query(Interaction.persona).distinct().count()
        
        return {
            "total_interactions": total_interactions,
            "unique_users": unique_users,
            "unique_personas": unique_personas,
            "database_path": DATABASE_URL
        }


if __name__ == "__main__":
    # Initialize database when run directly
    init_database()
    print("Database initialized successfully")
    print(f"Database location: {DATABASE_URL}")
    print("Database stats:", get_db_stats()) 