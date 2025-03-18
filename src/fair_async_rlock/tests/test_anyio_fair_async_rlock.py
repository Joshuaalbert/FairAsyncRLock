import asyncio
import multiprocessing
import random
from contextlib import suppress
from functools import wraps
from typing import Any, Awaitable, Callable, NoReturn, Union

import anyio
import anyio.lowlevel
import pytest

from fair_async_rlock import AnyIOFairAsyncRLock

pytestmark: pytest.MarkDecorator = pytest.mark.anyio

CoRo = Callable[..., Awaitable[Any]]

SMALL_DELAY = 0.04  # Just enough for python to reliably execute a few lines of code


class DummyError(Exception):
    pass


def with_timeout(t: float) -> Callable[[CoRo], CoRo]:
    def wrapper(corofunc: CoRo) -> CoRo:
        @wraps(corofunc)
        async def run(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            with anyio.move_on_after(t) as scope:
                await corofunc(*args, **kwargs)
            if scope.cancelled_caught:
                pytest.fail("Test timeout.")

        return run

    return wrapper


def repeat(n: int) -> Callable[[CoRo], CoRo]:
    def wrapper(corofunc: CoRo) -> CoRo:
        @wraps(corofunc)
        async def run(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            for _ in range(n):
                await corofunc(*args, **kwargs)

        return run

    return wrapper


@repeat(10)
@with_timeout(1)
async def test_reentrant() -> None:
    """Test that the lock can be acquired multiple times by the same task."""
    lock = AnyIOFairAsyncRLock()
    async with lock and lock:
        assert True


@repeat(10)
@with_timeout(1)
async def test_exclusion() -> None:
    """Test that the lock prevents multiple tasks from acquiring it at the same time."""
    lock = AnyIOFairAsyncRLock()
    got_in = anyio.Event()

    async def inner() -> None:
        async with lock:
            got_in.set()  # Never reached: Shouldn't happen

    async with lock, anyio.create_task_group() as tg:
        tg.start_soon(inner)
        await anyio.sleep(SMALL_DELAY)
        assert not got_in.is_set()
        tg.cancel_scope.cancel()


@repeat(10)
@with_timeout(1)
async def test_fairness() -> None:
    """Test that the lock is acquired in the order it is requested."""
    lock = AnyIOFairAsyncRLock()
    order: list[int] = []

    async def worker(n: int) -> None:
        async with lock:
            await anyio.sleep(SMALL_DELAY)
            order.append(n)

    async with anyio.create_task_group() as tg:
        for i in range(5):
            tg.start_soon(worker, i)
            await anyio.lowlevel.checkpoint()

    assert order == list(range(5))
    assert not lock.locked()


@repeat(10)
@with_timeout(1)
async def test_unowned_release() -> None:
    """Test that releasing an un-acquired lock is handled gracefully."""
    async with anyio.create_task_group() as tg:
        lock = AnyIOFairAsyncRLock()

        with pytest.raises(RuntimeError, match="Cannot release un-acquired lock."):
            lock.release()

        async def worker() -> None:
            with pytest.raises(RuntimeError, match="Cannot release un-acquired lock."):
                lock.release()

        tg.start_soon(worker)


@with_timeout(1)
async def test_stress_1() -> None:
    """Test that the lock can be acquired and released by multiple tasks rapidly."""
    lock = AnyIOFairAsyncRLock()
    num_tasks = 100
    iterations = 100

    async def worker() -> None:
        for _ in range(iterations):
            async with lock:
                pass

    async with anyio.create_task_group() as tg:
        for _ in range(num_tasks):
            tg.start_soon(worker)

    assert not lock.locked()


@with_timeout(1)
async def test_stress_2() -> None:
    """Test that the lock can be acquired and released by multiple tasks rapidly."""
    lock = AnyIOFairAsyncRLock()
    num_tasks = 100

    alive_tasks: int = 0
    async with anyio.create_task_group() as tg:

        async def worker() -> None:
            nonlocal alive_tasks
            alive_tasks += 1
            with anyio.CancelScope() as scope:
                while not scope.cancel_called:
                    async with lock:
                        n: int = random.randint(0, 2)  # noqa: S311
                        if n == 0:  # Create a new task 1/3 times.
                            tg.start_soon(worker)
                        else:  # Cancel a task 2/3 times.
                            scope.cancel()
            alive_tasks -= 1

        for _ in range(num_tasks):
            tg.start_soon(worker)

    assert alive_tasks == 0
    assert not lock.locked()


@repeat(10)
@with_timeout(1)
async def test_lock_status_checks() -> None:
    """Test that the lock status checks work as expected."""
    lock = AnyIOFairAsyncRLock()
    assert not lock.is_owner()
    async with lock:
        assert lock.is_owner()


@repeat(10)
@with_timeout(1)
async def test_nested_lock_acquisition() -> None:
    """Test that lock ownership is correctly tracked."""
    lock1 = AnyIOFairAsyncRLock()
    lock2 = AnyIOFairAsyncRLock()

    lock1_acquired = anyio.Event()
    worker_task: Union[anyio.TaskInfo, None] = None

    async def worker() -> None:
        nonlocal worker_task
        worker_task = anyio.get_current_task()
        async with lock1:
            lock1_acquired.set()
            await anyio.sleep(SMALL_DELAY)

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


@repeat(10)
@with_timeout(1)
async def test_lock_released_on_exception() -> None:
    """Test that the lock is released when an exception is raised."""
    lock = AnyIOFairAsyncRLock()
    with suppress(Exception):
        async with lock:
            raise DummyError

    assert not lock.locked()


@repeat(10)
@with_timeout(1)
async def test_release_foreign_lock() -> None:
    """Test that releasing a lock acquired by another task is handled gracefully."""
    lock = AnyIOFairAsyncRLock()
    lock_acquired = anyio.Event()

    async def task1() -> None:
        async with lock:
            lock_acquired.set()
            await anyio.sleep(SMALL_DELAY)

    async def task2() -> None:
        await lock_acquired.wait()
        with pytest.raises(RuntimeError, match="Cannot release foreign lock."):
            lock.release()

    async with anyio.create_task_group() as tg:
        tg.start_soon(task1)
        await lock_acquired.wait()
        tg.start_soon(task2)

    assert not lock.locked()


@repeat(10)
@with_timeout(1)
async def test_acquire_exception_handling() -> None:
    """Test that if an exception is raised by current owner during lock acquisition, the lock is still handed over."""
    lock = AnyIOFairAsyncRLock()
    lock_acquired = anyio.Event()
    success_flag = anyio.Event()

    async def failing_task() -> NoReturn:
        try:
            await lock.acquire()
            lock_acquired.set()
            await anyio.sleep(SMALL_DELAY)
            raise DummyError  # noqa: TRY301
        except DummyError:
            lock.release()

    async def succeeding_task() -> None:
        await lock.acquire()
        success_flag.set()
        lock.release()

    async with anyio.create_task_group() as tg:
        tg.start_soon(failing_task)
        await lock_acquired.wait()
        tg.start_soon(succeeding_task)

    assert not lock.locked()
    assert success_flag.is_set()


@repeat(10)
@with_timeout(1)
async def test_lock_cancellation_during_acquisition() -> None:
    """Test that if cancellation is raised during lock acquisition, the lock is not acquired."""
    lock = AnyIOFairAsyncRLock()
    t1_ac = anyio.Event()
    t2_ac = anyio.Event()
    t2_started = anyio.Event()

    async def task1() -> None:
        async with lock:
            t1_ac.set()
            await anyio.sleep(100)

    async def task2() -> None:
        await t1_ac.wait()
        t2_started.set()
        async with lock:
            t2_ac.set()  # Never reached: Shouldn't happen

    async with anyio.create_task_group() as tg:
        tg.start_soon(task1)
        tg.start_soon(task2)
        await t2_started.wait()
        tg.cancel_scope.cancel()

    assert t2_started.is_set()
    assert not t2_ac.is_set()
    assert not lock.locked()


@repeat(10)
@with_timeout(1)
async def test_lock_cancellation_after_acquisition() -> None:
    """Test that if cancellation is raised after lock acquisition, the lock is still released."""
    lock = AnyIOFairAsyncRLock()
    lock_acquired = anyio.Event()
    cancellation_event = anyio.Event()

    async def task_to_cancel() -> None:
        async with lock:
            lock_acquired.set()
            try:
                await anyio.sleep(SMALL_DELAY)
            except anyio.get_cancelled_exc_class():
                cancellation_event.set()

    async with anyio.create_task_group() as tg:
        tg.start_soon(task_to_cancel)
        await lock_acquired.wait()
        tg.cancel_scope.cancel()

    await cancellation_event.wait()

    assert not lock.locked()


def test_non_cooperative_cancel():
    def run_coroutine(q):
        async def run_test():
            lock = AnyIOFairAsyncRLock()

            async def while_loop():
                while True:
                    await lock.acquire()
                    # await anyio.sleep(0.) # Uncommenting makes the task cooperative

            async with anyio.create_task_group() as tg:
                tg.start_soon(while_loop)
                await anyio.lowlevel.checkpoint()
                tg.cancel_scope.cancel()
            assert lock._count == 0  # The lock should be released after the task is cancelled

        try:
            asyncio.run(run_test())
        except BaseException as e:
            q.put(e)

    # Because cancellation is cooperative if we try to use wait_for to test for timeout it will just run forever,
    # because control never yielded. await keyword does not yield control.
    # Thus, we must wrap into a process to test for hanging.
    q = multiprocessing.Queue()
    proc = multiprocessing.Process(target=run_coroutine, args=(q,))
    proc.start()
    # Wait for the process to finish, with a timeout.
    proc.join(timeout=2)

    # If the process is still alive, it means the coroutine did not yield and cancel.
    try:
        assert not proc.is_alive(), "Test did not terminate as expected."
        if not q.empty():
            raise q.get()
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=1)


def test_non_cooperative_cancel_reentrant():
    def run_coroutine(q):
        async def run_test():
            event = anyio.Event()
            lock = AnyIOFairAsyncRLock()

            async def while_loop():
                idx = 0
                while True:
                    await lock.acquire()
                    # await anyio.sleep(0.) # Uncommenting makes the task cooperative
                    idx += 1
                    if idx == 2:
                        event.set()

            async with anyio.create_task_group() as tg:
                tg.start_soon(while_loop)
                await anyio.lowlevel.checkpoint()
                await event.wait()
                tg.cancel_scope.cancel()
            assert lock._owner == None
            assert lock._count == 0  # The lock should be released after the task is cancelled

        try:
            asyncio.run(run_test())
        except BaseException as e:
            q.put(e)

    # Because cancellation is cooperative if we try to use wait_for to test for timeout it will just run forever,
    # because control never yielded. await keyword does not yield control.
    # Thus, we must wrap into a process to test for hanging.
    q = multiprocessing.Queue()
    proc = multiprocessing.Process(target=run_coroutine, args=(q,))
    proc.start()
    # Wait for the process to finish, with a timeout.
    proc.join(timeout=2)

    # If the process is still alive, it means the coroutine did not yield and cancel.
    try:
        assert not proc.is_alive(), "Test did not terminate as expected."
        if not q.empty():
            raise q.get()
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=1)


def test_non_cooperative_cancel_reentrant_nested():
    num_acquires = 2
    def run_coroutine(q):
        async def run_test():
            acquire_event_inner = anyio.Event()
            acquire_event_outer = anyio.Event()
            lock = AnyIOFairAsyncRLock()

            async def while_loop_inner():
                idx = 0
                while True:
                    if idx < num_acquires:
                        await lock.acquire()
                        # await asyncio.sleep(0) # Uncommenting makes the task cooperative
                        idx += 1
                        continue
                    else:
                        if not acquire_event_inner.is_set():
                            acquire_event_inner.set()

            async def while_loop_outer():
                idx = 0
                while True:
                    if idx < num_acquires:
                        await lock.acquire()
                        # await asyncio.sleep(0) # Uncommenting makes the task cooperative
                        idx += 1
                        continue
                    else:
                        if not acquire_event_outer.is_set():
                            async with anyio.create_task_group() as tg:
                                tg.start_soon(while_loop_inner)
                                acquire_event_outer.set()
                                await anyio.lowlevel.checkpoint()

            async with anyio.create_task_group() as tg:
                tg.start_soon(while_loop_outer)
                await anyio.lowlevel.checkpoint()
                await acquire_event_outer.wait()
                tg.cancel_scope.cancel()

            assert lock._owner == None
            assert lock._count == 0  # The lock should be released after the task is cancelled

        try:
            asyncio.run(run_test())
        except BaseException as e:
            q.put(e)

    # Because cancellation is cooperative if we try to use wait_for to test for timeout it will just run forever,
    # because control never yielded. await keyword does not yield control.
    # Thus, we must wrap into a process to test for hanging.
    q = multiprocessing.Queue()
    proc = multiprocessing.Process(target=run_coroutine, args=(q,))
    proc.start()
    # Wait for the process to finish, with a timeout.
    proc.join(timeout=2)

    # If the process is still alive, it means the coroutine did not yield and cancel.
    try:
        assert not proc.is_alive(), "Test did not terminate as expected."
        if not q.empty():
            raise q.get()
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=1)