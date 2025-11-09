"""
Useful decorators for various purposes.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable

from faex.utils.typing import is_hashable


def keyword_only_method(func: Callable) -> Callable:
    """
    Enforce that instance methods are called with keyword-only args.
    Also updates the visible signature to show keyword-only params.
    """
    sig = inspect.signature(func)
    params = list(sig.parameters.values())
    if not params:
        raise TypeError(f"{func.__name__} must be a method with 'self' first")

    # Create a public-facing signature where every param after self is KEYWORD_ONLY
    new_params = [params[0]] + [p.replace(kind=inspect.Parameter.KEYWORD_ONLY) for p in params[1:]]
    public_sig = sig.replace(parameters=new_params)

    @functools.wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if args:
            raise TypeError(f"{func.__name__}() accepts only keyword arguments")
        # Validate kwargs (presence, types, defaults) against the public signature
        public_sig.bind(self, **kwargs)
        return func(self, **kwargs)

    # Make tools/IDEs show the keyword-only signature
    wrapper.__signature__ = public_sig  # type: ignore[attr-defined]

    return wrapper


def cache_method(func: Callable) -> Callable:
    """
    Per-instance memoization for instance methods.

    For those calls with and without keywords, the cache key differs.

    Implementation Note:
        Works best when arguments are hashable. If not, we fall back to a stringified key (slower but safe).
    """
    cache_attr = f"__cache_{func.__name__}"

    @functools.wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        # get/create this method's cache for THIS instance
        cache = self.__dict__.setdefault(cache_attr, {})

        key: Any = None
        # build a key: (args, sorted kwargs) — common, fast path
        if kwargs:
            key = (args, tuple(sorted(kwargs.items())))
        else:
            key = (args, None)

        # Check if hashable
        if not is_hashable(key):
            # unhashable in args/kwargs: fall back to a stable string key
            key = repr((args, sorted(kwargs.items())))

        # Lookup in cache
        if key in cache:
            return cache[key]

        # Not found: compute and store
        result = func(self, *args, **kwargs)
        cache[key] = result
        return result

    # expose a way to clear just this method's cache on an instance
    def cache_clear(inst: Any) -> None:
        inst.__dict__.get(cache_attr, {}).clear()

    # Set the cache_clear method on the wrapper
    wrapper.cache_clear = cache_clear  # type: ignore[attr-defined]

    return wrapper
