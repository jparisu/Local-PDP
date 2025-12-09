from __future__ import annotations

import threading
import logging
from typing import Generic, Optional, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class SingletonFactory(Generic[T]):
    """
    A generic factory that makes any class a singleton per process.
    """

    def __init__(self, cls: Type[T]) -> None:
        self._cls: Type[T] = cls
        self._instance: Optional[T] = None
        self._lock = threading.Lock()

        logger.debug(f"Singleton created for class {cls.__name__}")

    def __call__(self, *args, **kwargs) -> T:
        if self._instance is None:
            with self._lock:
                if self._instance is None:
                    self._instance = self._cls(*args, **kwargs)
        return self._instance

    def get_instance(self) -> Optional[T]:
        """
        Get the singleton instance if it exists, or create it.

        Returns:
            Optional[T]: The singleton instance or None if not created yet.
        """
        return self.__call__()
