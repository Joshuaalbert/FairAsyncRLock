import random
from functools import wraps
from time import perf_counter_ns
from typing import Any, Awaitable, Callable, NoReturn

import anyio
import pytest

from fair_async_rlock import FairAsyncRLock

pytestmark: pytest.MarkDecorator = pytest.mark.anyio


def with_timeout(t: float):
    def wrapper(corofunc: Callable[[], Awaitable[Any]]):
        @wraps(corofunc)
        async def run(*args: Any, **kwargs: Any) -> Any:
            with anyio.move_on_after(t) as scope:
                await corofunc(*args, **kwargs)
            if scope.cancelled_caught:
                pytest.fail("Test timeout.")

        return run

    return wrapper


async def yield_control() -> None:
    # https://stackoverflow.com/a/74498442
    await anyio.sleep(0)
    await anyio.sleep(0)
    await anyio.sleep(0)


@with_timeout(1)
async def test_reentrant() -> None:
    lock = FairAsyncRLock()
    async with lock:
        async with lock:
            assert True


@with_timeout(1)
async def test_exclusion() -> None:
    lock = FairAsyncRLock()
    started = anyio.Event()
    got_in = anyio.Event()

    async def inner() -> None:
        started.set()
        async with lock:
            got_in.set()

    # Acquire the lock, then run the inner task. It shouldn't be able to acquire the lock.
    async with lock:
        async with anyio.create_task_group() as tg:
            tg.start_soon(inner)
            await yield_control()
            assert started.is_set()
            assert not got_in.is_set()
            tg.cancel_scope.cancel()


@with_timeout(1)
async def test_fairness() -> None:
    lock = FairAsyncRLock()
    order: list[int] = []

    async def worker(n: int) -> None:
        async with lock:
            await anyio.sleep(0.1)
            order.append(n)

    # Start several tasks to acquire the lock
    async with anyio.create_task_group() as tg:
        for i in range(5):
            tg.start_soon(worker, i)
            await yield_control()

    assert order == list(range(5))  # The tasks should have run in order
    assert not lock.locked()


@with_timeout(1)
async def test_unowned_release() -> None:
    async with anyio.create_task_group() as tg:
        lock = FairAsyncRLock()

        with pytest.raises(RuntimeError, match="Cannot release un-acquired lock."):
            lock.release()

        async def worker() -> None:
            with pytest.raises(RuntimeError, match="Cannot release un-acquired lock."):
                lock.release()

        tg.start_soon(worker)
        await yield_control()


@with_timeout(1)
async def test_performance() -> None:
    # This test is useful for measuring the overhead of the locking mechanism and can help determine whether it's
    # suitable for high-concurrency scenarios.
    lock = FairAsyncRLock()
    num_tasks = 1000
    order: list[int] = []

    async def worker(n: int) -> None:
        async with lock:
            order.append(n)

    async with anyio.create_task_group() as tg:
        for i in range(num_tasks):
            tg.start_soon(worker, i)
        start: int = perf_counter_ns()
    end: int = perf_counter_ns()

    print(f"Time to complete {num_tasks} tasks: {end - start}ns")
    assert (
        len(order) == num_tasks
    )  # AnyIO start_soon tasks are not guaranteed to start in order, but still make sure they did run.


@with_timeout(1)
async def test_stress():
    # We'll create a large number of tasks that all try to acquire and release the lock repeatedly.
    # This can help identify any issues that only occur under high load or after many operations.
    lock = FairAsyncRLock()
    num_tasks = 100
    iterations = 1000

    async def worker(n: int):
        for _ in range(iterations):
            async with lock:
                pass

    async with anyio.create_task_group() as tg:
        for i in range(num_tasks):
            tg.start_soon(worker, i)

    assert not lock.locked()


@with_timeout(1)
async def test_hard():
    # "Hard" Test: We'll create a scenario where tasks are constantly being created and cancelled,
    # while trying to acquire the lock. This can help identify any issues related to task cancellation and cleanup.
    lock = FairAsyncRLock()
    num_tasks = 1000

    alive_tasks: int = 0
    async with anyio.create_task_group() as tg:

        async def worker() -> None:
            nonlocal alive_tasks
            alive_tasks += 1
            with anyio.CancelScope() as scope:
                while not scope.cancel_called:
                    async with lock:
                        n: int = random.randint(0, 2)
                        if n == 0:  # Create a new task 1/3 times.
                            tg.start_soon(worker)
                        else:  # Cancel a task 2/3 times.
                            scope.cancel()
                    await yield_control()
            alive_tasks -= 1

        for _ in range(num_tasks):
            tg.start_soon(worker)

    assert alive_tasks == 0
    assert not lock.locked()


@with_timeout(1)
async def test_lock_status_checks() -> None:
    # We should add tests to validate the is_owner method in the FairAsyncRLock class.
    # This method is crucial as it determines whether a lock can be acquired or released by the current task.
    lock = FairAsyncRLock()

    assert not lock.is_owner()

    async with lock:
        assert lock.is_owner()


@with_timeout(1)
async def test_nested_lock_acquisition() -> None:
    # While reentrancy was tested, it was not tested in a nested scenario involving more than one task.
    # We can design a test case where multiple tasks try to acquire a lock which is already owned by a task
    # that is itself waiting for another lock. This tests the behavior of the FairAsyncRLock in nested lock
    # acquisition scenarios.
    lock1 = FairAsyncRLock()
    lock2 = FairAsyncRLock()

    lock1_acquired = anyio.Event()
    worker_task: anyio.TaskInfo | None = None

    async def worker() -> None:
        nonlocal worker_task
        worker_task = anyio.get_current_task()
        async with lock1:
            lock1_acquired.set()
            await yield_control()  # Yield control while holding lock1

    async def control_task() -> None:
        nonlocal worker_task
        async with anyio.create_task_group() as tg:
            tg.start_soon(worker)
            await lock1_acquired.wait()
            assert lock1.is_owner(task=worker_task)
            assert not lock2.is_owner()
            assert worker_task != anyio.get_current_task()
            async with lock2:
                assert lock1.is_owner(task=worker_task)
                assert lock2.is_owner()

    await control_task()


@with_timeout(10)
async def test_performance_comparison() -> None:
    fair_lock = FairAsyncRLock()
    stdlib_lock = anyio.Lock()
    num_tasks = 10000  # TODO: Bump back up to 100000

    async def worker(lock: anyio.Lock | FairAsyncRLock) -> None:
        async with lock:
            await yield_control()

    # Measure performance of FairAsyncRLock
    async with anyio.create_task_group() as tg:
        for _ in range(num_tasks):
            tg.start_soon(worker, fair_lock)
        start_fair: int = perf_counter_ns()
    end_fair: int = perf_counter_ns()
    duration_fair: int = end_fair - start_fair

    # Measure performance of anyio.Lock
    async with anyio.create_task_group() as tg:
        for _ in range(num_tasks):
            tg.start_soon(worker, stdlib_lock)
        start_stdlib: int = perf_counter_ns()
    end_stdlib: int = perf_counter_ns()
    duration_stdlib: int = end_stdlib - start_stdlib

    print(f"Time to complete {num_tasks} tasks with FairAsyncRLock: {duration_fair}ns")
    print(f"Time to complete {num_tasks} tasks with anyio.Lock: {duration_stdlib}ns")
    # We find that it's about the same performance as anyio.Lock.
    perf_ratio: float = duration_fair / duration_stdlib
    if perf_ratio > 1:
        print(f"Relative performance: {(perf_ratio - 1) * 100:0.1f}% slower")
    else:
        print(f"Relative performance: {(1 - perf_ratio) * 100:0.1f}% faster")
    assert perf_ratio < 2.0  # Solid upper bound


@with_timeout(1)
async def test_lock_released_on_exception() -> None:
    lock = FairAsyncRLock()
    with pytest.raises(Exception):
        async with lock:
            raise Exception("Test")

    assert not lock.locked()


@with_timeout(1)
async def test_release_foreign_lock() -> None:
    lock = FairAsyncRLock()
    lock_acquired = anyio.Event()

    async def task1() -> None:
        async with lock:
            lock_acquired.set()
            await anyio.sleep(0.1)

    async def task2() -> None:
        await lock_acquired.wait()
        try:
            lock.release()
        except RuntimeError as e:
            assert str(e).startswith("Cannot release foreign lock.")
            return

    async with anyio.create_task_group() as tg:
        tg.start_soon(task1)
        await yield_control()
        tg.start_soon(task2)

    assert not lock.locked()


@with_timeout(1)
async def test_lock_acquired_released_normally() -> None:
    lock = FairAsyncRLock()
    async with lock:
        assert lock._count == 1  # pyright: ignore[reportPrivateUsage]
        assert lock._owner is not None  # pyright: ignore[reportPrivateUsage]
        assert lock._owner == anyio.get_current_task()  # pyright: ignore[reportPrivateUsage]

    assert lock._owner is None  # pyright: ignore[reportPrivateUsage]
    assert lock._count == 0  # pyright: ignore[reportPrivateUsage]


@with_timeout(1)
async def test_acquire_exception_handling() -> None:
    # We can simulate an exception occurring in the acquire() method and validate that it does not leave the
    # lock in an inconsistent state.
    lock = FairAsyncRLock()

    async def failing_task() -> NoReturn:
        try:
            await lock.acquire()
            raise RuntimeError("Simulated exception during acquire")
        except RuntimeError:
            lock.release()

    async def succeeding_task() -> None:
        await lock.acquire()
        lock.release()

    async with anyio.create_task_group() as tg:
        tg.start_soon(failing_task)
        await yield_control()
        tg.start_soon(succeeding_task)

    assert not lock.locked()


@with_timeout(1)
async def test_lock_cancellation_during_acquisition() -> None:
    # We need to verify that if a task is cancelled while waiting for the lock, it gets removed from the queue.
    lock = FairAsyncRLock()
    t1_ac = anyio.Event()
    t2_ac = anyio.Event()
    t2_started = anyio.Event()

    async def task1() -> None:
        async with lock:
            t1_ac.set()
            await anyio.sleep(1)

    async def task2() -> None:
        await t1_ac.wait()
        t2_started.set()
        async with lock:
            t2_ac.set()

    async with anyio.create_task_group() as tg:
        tg.start_soon(task1)
        tg.start_soon(task2)
        await yield_control()

        tg.cancel_scope.cancel()
        await yield_control()

        assert t2_started.is_set()
        assert not t2_ac.is_set()

    assert not lock.locked()


@with_timeout(1)
async def test_lock_cancellation_after_acquisition() -> None:
    lock = FairAsyncRLock()
    cancellation_event = anyio.Event()

    async def task_to_cancel() -> None:
        async with lock:
            try:
                await anyio.sleep(1)
            except anyio.get_cancelled_exc_class():
                cancellation_event.set()

    async with anyio.create_task_group() as tg:
        tg.start_soon(task_to_cancel)
        await yield_control()
        tg.cancel_scope.cancel()

    await cancellation_event.wait()

    assert not lock.locked()
