from abc import ABC, abstractmethod
from collections import deque

from typing import Any, Generic, TypeVar, Protocol

class _Event(Protocol):
    def set(self) -> None: ...
    async def wait(self) -> Any: ...


TaskType = TypeVar("TaskType")
EventType = TypeVar("EventType", bound=_Event)


class AbstractFairAsyncRLock(ABC, Generic[TaskType, EventType]):
    @abstractmethod
    def _get_current_task(self) -> TaskType | None:
        ...

    @abstractmethod
    def _get_cancelled_exc_class(self) -> type[BaseException]:
        ...

    @abstractmethod
    def _get_wake_event(self) -> EventType:
        ...

    @abstractmethod
    async def _checkpoint(self) -> None:
        ...


class BaseFairAsyncRLock(AbstractFairAsyncRLock[TaskType, EventType]):
    """
    A fair reentrant lock for async programming. Fair means that it respects the order of acquisition.
    """

    def __init__(self):
        self._owner: TaskType | None = None
        self._count: int = 0
        self._owner_transfer: bool = False
        self._queue: deque[EventType] = deque()

    def is_owner(self, task: TaskType | None = None) -> bool:
        if task is None:
            task = self._get_current_task()
        return self._owner == task

    def locked(self) -> bool:
        return self._owner is not None

    async def acquire(self) -> None:
        """Acquire the lock."""
        me = self._get_current_task()

        # If the lock is reentrant, acquire it immediately
        if self.is_owner(task=me):
            self._count += 1
            try:
                await self._checkpoint()
            except self._get_cancelled_exc_class():
                # Cancelled, while reentrant, so release the lock
                self._owner_transfer = False
                self._owner = me
                self._count = 1
                self._current_task_release()
                raise
            return

        # If the lock is free (and ownership not in midst of transfer), acquire it immediately
        if self._count == 0 and not self._owner_transfer:
            self._owner = me
            self._count = 1
            try:
                await self._checkpoint()
            except self._get_cancelled_exc_class():
                self._current_task_release()
                raise
            return

        # Create an event for this task, to notify when it's ready for acquire
        event = self._get_wake_event()
        self._queue.append(event)

        # Wait for the lock to be free, then acquire
        try:
            await event.wait()
            self._owner_transfer = False
            self._owner = me
            self._count = 1
        except self._get_cancelled_exc_class():
            try:  # if in queue, then cancelled before release
                self._queue.remove(event)
            except ValueError:
                # otherwise, release happened, this was next, and we simulate passing on
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
                event = self._queue.popleft()
                event.set()
                # Setting this here prevents another task getting lock until owner transfer.
                self._owner_transfer = True

    def release(self):
        """Release the lock"""
        me = self._get_current_task()

        if self._owner is None:
            raise RuntimeError(f"Cannot release un-acquired lock. {me} tried to release.")

        if not self.is_owner(task=me):
            raise RuntimeError(f"Cannot release foreign lock. {me} tried to unlock {self._owner}.")

        self._current_task_release()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.release()
