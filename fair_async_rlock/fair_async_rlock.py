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

        if self.is_owner(task=me):
            self._count += 1
            return

        # If the lock is free or reentrant, acquire it immediately
        if self._count == 0:
            # if self._count == 0  or self._owner == me: (redundant second clause)
            self._owner = me
            self._count += 1
        else:
            # Create an event for this task
            event = asyncio.Event()
            self._queue.append(event)

            # Wait for the lock to be free
            try:
                await event.wait()
            except asyncio.CancelledError:
                self._queue.remove(event)
                raise

            self._owner = me
            self._count = 1

    async def release(self):
        """Release the lock"""
        me = asyncio.current_task()

        if self._owner is None:
            raise RuntimeError("Cannot release un-acquired lock.")

        if not self.is_owner(task=me):
            raise RuntimeError("Cannot release foreign lock.")

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
