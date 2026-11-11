import logging
from urllib.parse import unquote, urlparse

import aiomysql

logger = logging.getLogger(__name__)


class QueryLogger:
    def __init__(self, mysql_url: str | None) -> None:
        self.mysql_url = mysql_url
        self.pool: aiomysql.Pool | None = None

    @property
    def enabled(self) -> bool:
        return self.mysql_url is not None

    async def open(self) -> None:
        if not self.mysql_url:
            return
        parsed = urlparse(self.mysql_url)
        if parsed.scheme not in {"mysql", "mysql+aiomysql"} or not parsed.hostname:
            raise ValueError("MYSQL_URL must use mysql:// or mysql+aiomysql://")
        self.pool = await aiomysql.create_pool(
            host=parsed.hostname,
            port=parsed.port or 3306,
            user=unquote(parsed.username or ""),
            password=unquote(parsed.password or ""),
            db=unquote(parsed.path.lstrip("/")),
            minsize=1,
            maxsize=5,
            autocommit=True,
        )
        async with self.pool.acquire() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """CREATE TABLE IF NOT EXISTS query_logs (
                        id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
                        question TEXT NOT NULL,
                        document_id CHAR(36) NULL,
                        latency_ms DOUBLE NOT NULL,
                        cached BOOLEAN NOT NULL DEFAULT FALSE,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        INDEX idx_query_logs_created_at (created_at)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"""
                )
        logger.info("MySQL query logging enabled")

    async def log(self, question: str, document_id: str | None, latency_ms: float, cached: bool) -> None:
        if not self.pool:
            return
        try:
            async with self.pool.acquire() as connection:
                async with connection.cursor() as cursor:
                    await cursor.execute(
                        "INSERT INTO query_logs (question, document_id, latency_ms, cached) "
                        "VALUES (%s, %s, %s, %s)",
                        (question, document_id, latency_ms, cached),
                    )
        except Exception:
            logger.exception("Failed to write query log")

    async def close(self) -> None:
        if self.pool:
            self.pool.close()
            await self.pool.wait_closed()
