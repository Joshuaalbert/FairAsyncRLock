import asyncio
from collections import deque

__all__ = ['FairAsyncRLock']

class FairAsyncRLock:
    """
    A fair reentrant lock for async programming. Fair means that it respects the order of acquisition.
    """

    def __init__(self):
        self._owner: asyncio.Task | None = None
        self._count = 0
        self._queue = deque()

    def is_owner(self, task=None):
        if task is None:
            task = asyncio.current_task()
        return self._owner == task

    async def acquire(self):
        """Acquire the lock."""
        me = asyncio.current_task()

        # If the lock is reentrant, acquire it immediately
        if self.is_owner(task=me):
            self._count += 1
            return

        # If the lock is free, acquire it immediately
        if self._count == 0:
            self._owner = me
            self._count = 1
            return

        # Create an event for this task, to notify when it's ready for acquire
        event = asyncio.Event()
        self._queue.append(event)

        # Wait for the lock to be free, then acquire
        try:
            await event.wait()
            self._owner = me
            self._count = 1
        except asyncio.CancelledError:
            self._queue.remove(event)
            raise

    async def release(self):
        """Release the lock"""
        me = asyncio.current_task()

        if self._owner is None:
            raise RuntimeError(f"Cannot release un-acquired lock. {me} tried to release.")

        if not self.is_owner(task=me):
            raise RuntimeError(f"Cannot release foreign lock. {me} tried to unlock {self._owner}.")

        self._count -= 1
        if self._count == 0:
            self._owner = None
            if self._queue:
                # Wake up the next task in the queue
                event = self._queue.popleft()
                event.set()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.release()
