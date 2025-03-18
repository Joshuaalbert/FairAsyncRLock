import asyncio
from typing import Any, TypeVar

from fair_async_rlock.base_fair_async_rlock import BaseFairAsyncRLock

__all__ = [
    'FairAsyncRLock'
]
TaskType = TypeVar('TaskType')
EventType = TypeVar('EventType')


class FairAsyncRLock(BaseFairAsyncRLock[asyncio.Task[Any], asyncio.Event]):
    """
    A fair reentrant lock for async programming. Fair means that it respects the order of acquisition.
    """

    def _get_current_task(self) -> asyncio.Task[Any] | None:
        return asyncio.current_task()

    def _get_cancelled_exc_class(self) -> type[BaseException]:
        return asyncio.CancelledError

    def _get_wake_event(self) -> asyncio.Event:
        return asyncio.Event()

    async def _checkpoint(self) -> None:
        await asyncio.sleep(0)
