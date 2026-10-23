
import os

from sqlalchemy import URL, create_engine, text


def get_database_engine():
    """
    Create a connection engine for the Vantara PostgreSQL database.
    """

    password = os.getenv("DB_PASSWORD")

    if not password:
        raise ValueError(
            "DB_PASSWORD environment variable is not set."
        )

    database_url = URL.create(
        drivername="postgresql+psycopg",
        username=os.getenv("DB_USER", "postgres"),
        password=password,
        host=os.getenv("DB_HOST", "127.0.0.1"),
        port=int(os.getenv("DB_PORT", "5432")),
        database=os.getenv("DB_NAME", "vantara_db"),
    )

    engine = create_engine(
        database_url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 5},
    )

    return engine


def test_database_connection():
    """
    Test PostgreSQL connectivity and return the connected database name.
    """

    engine = get_database_engine()

    try:
        with engine.connect() as connection:
            result = connection.execute(
                text("SELECT current_database();")
            )

            return result.scalar_one()

    finally:
        engine.dispose()
