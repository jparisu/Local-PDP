from __future__ import annotations
from typing import TypeVar, Generic, Type, Optional
import threading

T = TypeVar("T")

class SingletonFactory(Generic[T]):
    """
    A generic factory that makes any class a singleton per process.
    """
    def __init__(self, cls: Type[T]) -> None:
        self._cls: Type[T] = cls
        self._instance: Optional[T] = None
        self._lock = threading.Lock()

    def __call__(self, *args, **kwargs) -> T:
        if self._instance is None:
            with self._lock:
                if self._instance is None:
                    self._instance = self._cls(*args, **kwargs)
        return self._instance
