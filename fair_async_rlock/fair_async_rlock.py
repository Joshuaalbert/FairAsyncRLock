from collections import deque
from typing import Self

import anyio

__all__ = ["FairAsyncRLock"]


class FairAsyncRLock:
    """A fair reentrant lock for async programming. Fair means that it respects the order of acquisition."""

    def __init__(self) -> None:
        self._owner: anyio.TaskInfo | None = None
        self._count = 0
        self._owner_transfer = False
        self._queue: deque[anyio.Event] = deque()

    def is_owner(self, task: anyio.TaskInfo | None = None) -> bool:
        if task is None:
            task = anyio.get_current_task()
        return self._owner == task

    async def acquire(self) -> None:
        """Acquire the lock."""
        me = anyio.get_current_task()

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
        event = anyio.Event()
        self._queue.append(event)

        # Wait for the lock to be free, then acquire
        try:
            await event.wait()
            self._owner_transfer = False
            self._owner = me
            self._count = 1
        except anyio.get_cancelled_exc_class():
            try:  # if in queue, then cancelled before release
                self._queue.remove(event)
            except (
                ValueError
            ):  # otherwise, release happened, this was next, and we simulate passing on
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

    def release(self) -> None:
        """Release the lock."""
        me = anyio.get_current_task()

        if self._owner is None:
            msg = f"Cannot release un-acquired lock. {me} tried to release."
            raise RuntimeError(msg)

        if not self.is_owner(task=me):
            msg = f"Cannot release foreign lock. {me} tried to unlock {self._owner}."
            raise RuntimeError(msg)

        self._current_task_release()

    async def __aenter__(self) -> Self:
        await self.acquire()
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.release()

    def locked(self) -> bool:
        return self._count > 0
