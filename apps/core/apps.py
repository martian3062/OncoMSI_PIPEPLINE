from django.apps import AppConfig
from django.db.backends.signals import connection_created


class CoreConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.core"

    def ready(self) -> None:
        connection_created.connect(_configure_sqlite, dispatch_uid="apps.core.sqlite.pragmas")


def _configure_sqlite(sender, connection, **kwargs) -> None:
    if connection.vendor != "sqlite":
        return
    with connection.cursor() as cursor:
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA synchronous=NORMAL;")
        cursor.execute("PRAGMA busy_timeout=30000;")
