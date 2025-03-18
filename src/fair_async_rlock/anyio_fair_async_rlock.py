from typing import TypeVar

import anyio

from fair_async_rlock.base_fair_async_rlock import BaseFairAsyncRLock

__all__ = [
    'AnyIOFairAsyncRLock'
]
TaskType = TypeVar('TaskType')
EventType = TypeVar('EventType')


class AnyIOFairAsyncRLock(BaseFairAsyncRLock[anyio.TaskInfo, anyio.Event]):
    """
    A fair reentrant lock for async programming. Fair means that it respects the order of acquisition.
    """

    def _get_current_task(self) -> anyio.TaskInfo:
        return anyio.get_current_task()

    def _get_cancelled_exc_class(self) -> type[BaseException]:
        return anyio.get_cancelled_exc_class()

    def _get_wake_event(self) -> anyio.Event:
        return anyio.Event()

    async def _checkpoint(self) -> None:
        await anyio.lowlevel.checkpoint()
