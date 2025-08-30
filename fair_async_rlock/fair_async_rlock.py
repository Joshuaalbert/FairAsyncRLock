from __future__ import annotations
import asyncio
from collections import deque
from typing import Optional, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import TracebackType

__all__ = [
    'FairAsyncRLock'
]


class FairAsyncRLock:
    """
    A fair reentrant lock for async programming. Fair means that it respects the order of acquisition.
    """

    __slots__ = ("_owner", "_count", "_owner_transfer", "_queue", "_loop")

    def __init__(self) -> None:
        self._owner: asyncio.Task | None = None
        self._count = 0
        self._owner_transfer = False
        self._queue: deque[asyncio.Future[None]] = deque()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
    
    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        if not self._loop: 
            self._loop = asyncio.get_event_loop()
        return self._loop

    def is_owner(self, task:Optional[asyncio.Task[Any]] = None) -> bool:
        return self._owner == (task or asyncio.current_task())

    def locked(self) -> bool:
        """determines if the lock is being currently held or not"""
        return self._owner is not None

    async def acquire(self) -> None:
        """Acquire the lock."""
        me = asyncio.current_task()

        # If the lock is reentrant, acquire it immediately
        if self.is_owner(task=me):
            self._count += 1
            return

        # If the lock is free (and ownership not in midst of transfer), acquire it immediately
        if self._count == 0 and not self._owner_transfer:
            self._owner = me
            self._count = 1
            return

        # Create an event for this task, to notify when it's ready for acquire
        fut = self.loop.create_future()
        self._queue.append(fut)

        # Wait for the lock to be free, then acquire
        try:
            await fut
            self._owner_transfer = False
            self._owner = me
            self._count = 1
        except asyncio.CancelledError:
            try:  # if in queue, then cancelled before release
                self._queue.remove(fut)
            except ValueError:  # otherwise, release happened, this was next, and we simulate passing on
                self._owner_transfer = False
                self._owner = me
                self._count = 1
                self._current_task_release()
            raise

    def _current_task_release(self) -> None:
        self._count -= 1
        if self._count == 0:
            self._owner = None
            if self._queue:
                # Wake up the next task in the queue
                self._queue.popleft().set_result(None)
                # Setting this here prevents another task getting lock until owner transfer.
                self._owner_transfer = True

    def release(self) -> None:
        """Release the lock"""
        me = asyncio.current_task()

        if self._owner is None:
            raise RuntimeError(f"Cannot release un-acquired lock. {me} tried to release.")

        if not self.is_owner(task=me):
            raise RuntimeError(f"Cannot release foreign lock. {me} tried to unlock {self._owner}.")

        self._current_task_release()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(
        self, 
        exc_type: Optional[type[BaseException]], 
        exc:Optional[BaseException], 
        tb:Optional[TracebackType]
    ) -> None:
        self.release()
