import asyncio
import random
from time import monotonic_ns, perf_counter

import pytest

from fair_async_rlock import FairAsyncRLock


@pytest.mark.asyncio
async def test_reentrant():
    lock = FairAsyncRLock()
    async with lock:
        async with lock:  # This should not block
            assert True


@pytest.mark.asyncio
async def test_exclusion():
    lock = FairAsyncRLock()
    got_in = False

    async def inner():
        nonlocal got_in
        async with lock:
            got_in = True

    # Acquire the lock, then run the inner task. It shouldn't be able
    # to acquire the lock.
    async with lock:
        asyncio.create_task(inner())
        await asyncio.sleep(0)  # Give the inner task a chance to run
        assert not got_in


@pytest.mark.asyncio
async def test_fairness():
    lock = FairAsyncRLock()
    order = []

    async def worker(n):
        async with lock:
            order.append(n)

    # Start several tasks to acquire the lock
    tasks = [asyncio.create_task(worker(i)) for i in range(5)]

    # Make sure they all start and try to acquire the lock before releasing it
    await asyncio.sleep(0)

    async with lock:
        pass  # Release the lock

    await asyncio.gather(*tasks)

    assert order == list(range(5))  # The tasks should have run in order


@pytest.mark.asyncio
async def test_unowned_release():
    lock = FairAsyncRLock()

    with pytest.raises(RuntimeError, match="Cannot release un-acquired lock."):
        lock.release()

    async def worker():
        with pytest.raises(RuntimeError, match="Cannot release un-acquired lock."):
            lock.release()

    await asyncio.gather(worker())


@pytest.mark.asyncio
async def test_performance():
    # This test is useful for measuring the overhead of the locking mechanism and can help determine whether it's
    # suitable for high-concurrency scenarios.
    lock = FairAsyncRLock()
    num_tasks = 1000
    order = []

    async def worker(n):
        async with lock:
            order.append(n)

    tasks = [asyncio.create_task(worker(i)) for i in range(num_tasks)]

    start = monotonic_ns()
    await asyncio.gather(*tasks)
    end = monotonic_ns()

    print(f"Time to complete {num_tasks} tasks: {end - start} ns")
    assert order == list(range(num_tasks))  # The tasks should have run in order


@pytest.mark.asyncio
async def test_stress():
    # We'll create a large number of tasks that all try to acquire and release the lock repeatedly.
    # This can help identify any issues that only occur under high load or after many operations.
    lock = FairAsyncRLock()
    num_tasks = 100
    iterations = 1000

    async def worker(n):
        for _ in range(iterations):
            async with lock:
                pass

    tasks = [asyncio.create_task(worker(i)) for i in range(num_tasks)]

    await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_hard():
    # "Hard" Test: We'll create a scenario where tasks are constantly being created and cancelled,
    # while trying to acquire the lock. This can help identify any issues related to task cancellation and cleanup.
    lock = FairAsyncRLock()
    num_tasks = 100
    iterations = 1000

    async def worker(n):
        for _ in range(iterations):
            async with lock:
                if n % 10 == 0:  # Cancel every 10th task
                    raise asyncio.CancelledError

    tasks = [asyncio.create_task(worker(i)) for i in range(num_tasks)]

    with pytest.raises(asyncio.CancelledError):
        await asyncio.gather(*tasks)

    assert lock._count == 0  # The lock should be released after all tasks are done
    assert lock._owner is None


@pytest.mark.asyncio
async def test_lock_status_checks():
    # We should add tests to validate the is_owner method in the FairAsyncRLock class.
    # This method is crucial as it determines whether a lock can be acquired or released by the current task.
    lock = FairAsyncRLock()

    # The lock should not have an owner initially
    assert not lock.is_owner()

    # After acquiring the lock, it should be owned by the current task
    async with lock:
        assert lock.is_owner()


@pytest.mark.asyncio
async def test_nested_lock_acquisition():
    # While reentrancy was tested, it was not tested in a nested scenario involving more than one task.
    # We can design a test case where multiple tasks try to acquire a lock which is already owned by a task
    # that is itself waiting for another lock. This tests the behavior of the FairAsyncRLock in nested lock
    # acquisition scenarios.
    lock1 = FairAsyncRLock()
    lock2 = FairAsyncRLock()

    lock1_acquired = asyncio.Event()

    async def worker():
        async with lock1:
            lock1_acquired.set()  # Signal that lock1 has been acquired
            await asyncio.sleep(0)  # Yield control while holding lock1
        # At this point, lock1 is released

    async def control_task():
        task = asyncio.create_task(worker())
        await lock1_acquired.wait()  # Wait for worker to acquire lock1
        assert lock1.is_owner(task=task)  # worker task should own lock1
        async with lock2:  # Acquire lock2
            assert lock1.is_owner(task=task)  # worker task should still own lock1
        await task  # Await completion of worker task after lock2 is released

    await control_task()


@pytest.mark.asyncio
async def test_starvation():
    # While fairness was tested, starvation, where a low-priority task could potentially be waiting forever
    # while higher-priority tasks continuously acquire the lock, is not explicitly covered. The design of the
    # FairAsyncRLock should prevent this from happening, but it could be worthwhile to add a test case that
    # specifically tests for this condition.
    lock = FairAsyncRLock()
    order = []

    async def worker(n):
        async with lock:
            order.append(n)

    # Start a low-priority task
    low_priority_task = asyncio.create_task(worker(0))

    # Give it a moment to start
    await asyncio.sleep(0)

    # Start several high-priority tasks
    high_priority_tasks = [asyncio.create_task(worker(i)) for i in range(1, 10)]

    # Wait for all tasks to complete
    await low_priority_task
    await asyncio.gather(*high_priority_tasks)

    # Check that the low-priority task was able to acquire the lock
    assert 0 in order


@pytest.mark.asyncio
async def test_concurrent_acquisition():
    lock = FairAsyncRLock()
    result = []

    async def worker(n):
        await lock.acquire()  # This will block until the lock can be acquired
        result.append(n)
        await asyncio.sleep(0)  # Yield control
        lock.release()

    # Start several tasks concurrently
    tasks = [asyncio.create_task(worker(i)) for i in range(5)]

    await asyncio.gather(*tasks)

    # All tasks should have been able to acquire the lock, but only one at a time
    assert len(result) == 5


@pytest.mark.asyncio
async def test_performance_comparison():
    fair_lock = FairAsyncRLock()
    asyncio_lock = asyncio.Lock()
    num_tasks = 100000

    async def worker(lock):
        async with lock:
            await asyncio.sleep(0)  # Simulate some work

    # Measure performance of FairAsyncRLock
    fair_tasks = [asyncio.create_task(worker(fair_lock)) for _ in range(num_tasks)]
    start_fair = perf_counter()
    await asyncio.gather(*fair_tasks)
    duration_fair = perf_counter() - start_fair

    # Measure performance of asyncio.Lock
    asyncio_tasks = [asyncio.create_task(worker(asyncio_lock)) for _ in range(num_tasks)]
    start_asyncio = perf_counter()
    await asyncio.gather(*asyncio_tasks)
    duration_asyncio = perf_counter() - start_asyncio

    print(f"Time to complete {num_tasks} tasks with FairAsyncRLock: {duration_fair} seconds")
    print(f"Time to complete {num_tasks} tasks with asyncio.Lock: {duration_asyncio} seconds")
    # We find that it's about the same performance as asyncio.Lock.
    perf_ratio = duration_fair / duration_asyncio
    if perf_ratio > 1:
        print(f"Relative performance: {(perf_ratio - 1) * 100:0.1f}% slower")
    else:
        print(f"Relative performance: {(1 - perf_ratio) * 100:0.1f}% faster")
    assert perf_ratio < 2.  # Solid upper bound


@pytest.mark.asyncio
async def test_lock_released_on_exception():
    lock = FairAsyncRLock()
    with pytest.raises(Exception):
        async with lock:
            raise Exception("Test")
    assert lock._count == 0
    assert lock._owner is None


@pytest.mark.asyncio
async def test_release_foreign_lock():
    lock = FairAsyncRLock()

    async def task1():
        async with lock:
            await asyncio.sleep(0.1)  # Sleep to ensure that task2 gets to the point where it's waiting for the lock

    async def task2():
        # Wait for both tasks to complete
        try:
            lock.release()
        except RuntimeError as e:
            assert str(e).startswith("Cannot release foreign lock.")
            return

    # Create the tasks and schedule them
    task1_handle = asyncio.create_task(task1())
    task2_handle = asyncio.create_task(task2())

    # Wait for both tasks to complete
    await asyncio.gather(task1_handle, task2_handle)


@pytest.mark.asyncio
async def test_lock_acquired_released_normally():
    lock = FairAsyncRLock()
    async with lock:
        assert lock._count == 1
        assert lock._owner is not None
        assert lock._owner == asyncio.current_task()
    assert lock._owner is None
    assert lock._count == 0


@pytest.mark.asyncio
async def test_async_release():
    # This test checks if the release() method works correctly when turned into an async function.
    # It creates two tasks that sequentially acquire and release the lock, ensuring that the second task can
    # acquire the lock after the first one has released it.
    lock = FairAsyncRLock()

    async def task1():
        async with lock:
            await asyncio.sleep(0.1)

    async def task2():
        async with lock:
            pass

    task1 = asyncio.create_task(task1())
    task2 = asyncio.create_task(task2())

    await asyncio.gather(task1, task2)

    # Ensure that lock is not owned and queue is empty after tasks are done
    assert lock._owner is None
    assert len(lock._queue) == 0


@pytest.mark.asyncio
async def test_acquire_exception_handling():
    # We can simulate an exception occurring in the acquire() method and validate that it does not leave the
    # lock in an inconsistent state.
    lock = FairAsyncRLock()

    async def failing_task():
        try:
            await lock.acquire()
            raise RuntimeError("Simulated exception during acquire")
        except:
            lock.release()
            raise

    async def succeeding_task():
        await lock.acquire()
        lock.release()

    task1 = asyncio.create_task(failing_task())
    task2 = asyncio.create_task(succeeding_task())
    with pytest.raises(RuntimeError, match="Simulated exception during acquire"):
        await asyncio.gather(task1, task2)

    # Ensure that lock is not owned and queue is empty after exception
    assert lock._owner is None
    assert len(lock._queue) == 0


@pytest.mark.asyncio
async def test_task_cancellation():
    # We need to verify that if a task is cancelled while waiting for the lock, it gets removed from the queue.
    lock = FairAsyncRLock()
    t1_ac = asyncio.Event()
    t1_done = asyncio.Event()
    t2_ac = asyncio.Event()

    async def task1():
        async with lock:
            t1_ac.set()
            await t1_done.wait()

    async def task2():
        await t1_ac.wait()
        async with lock:
            t2_ac.set()
            await asyncio.sleep(1.)  # Let's ensure the lock is held for a bit

    task1 = asyncio.create_task(task1())
    task2 = asyncio.create_task(task2())
    await asyncio.sleep(0.1)  # Yield control to allow tasks to start
    task2.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task2
    assert not t2_ac.is_set()  # shouldn't acquire
    t1_done.set()  # Let T1 finish
    await task1  # Ensure task1 has a chance to release the lock
    # Ensure that lock is not owned and queue is empty after cancellation
    assert lock._owner is None
    assert len(lock._queue) == 0


@pytest.mark.asyncio
async def test_lock_cancellation_before_acquisition():
    lock = FairAsyncRLock()
    cancellation_event = asyncio.Event()

    async def task_to_cancel():
        try:
            async with lock:
                await asyncio.sleep(1)  # simulate some work
        except asyncio.CancelledError:
            cancellation_event.set()

    task = asyncio.create_task(task_to_cancel())
    await asyncio.sleep(0)  # yield control to let the task start
    task.cancel()
    await cancellation_event.wait()  # wait for the task to handle the cancellation

    assert lock._owner is None  # lock should not be owned by any task


@pytest.mark.asyncio
async def test_lock_cancellation_during_acquisition():
    lock = FairAsyncRLock()
    acquisition_event = asyncio.Event()
    cancellation_event = asyncio.Event()

    async def task_acquiring_lock():
        await lock.acquire()  # acquire the lock without releasing
        acquisition_event.set()  # signal that lock has been acquired

    async def task_to_cancel():
        await acquisition_event.wait()  # wait for the other task to acquire the lock
        try:
            async with lock:  # attempt to acquire the lock
                await asyncio.sleep(1)  # simulate some work
        except asyncio.CancelledError:
            cancellation_event.set()

    first_task = asyncio.create_task(task_acquiring_lock())
    task = asyncio.create_task(task_to_cancel())
    await asyncio.sleep(0)  # yield control to let the tasks start
    await acquisition_event.wait()  # wait for the lock to be acquired
    task.cancel()
    await cancellation_event.wait()  # wait for the task to handle the cancellation

    assert lock.is_owner(task=first_task)  # lock should still be owned by the first task


@pytest.mark.asyncio
async def test_lock_cancellation_after_acquisition():
    lock = FairAsyncRLock()
    cancellation_event = asyncio.Event()

    async def task_to_cancel():
        async with lock:  # acquire the lock
            try:
                await asyncio.sleep(1)  # simulate some work
            except asyncio.CancelledError:
                cancellation_event.set()

    task = asyncio.create_task(task_to_cancel())
    await asyncio.sleep(0)  # yield control to let the task start
    task.cancel()
    await cancellation_event.wait()  # wait for the task to handle the cancellation

    assert lock._owner is None  # lock should not be owned by any task


@pytest.mark.asyncio
async def test_stochastic_cancellation():
    lock = FairAsyncRLock()
    num_tasks = 100  # number of tasks to create
    tasks = []
    cancellation_occurred = asyncio.Event()

    async def task_func(task_id):
        """Function to be run in tasks. Tries to acquire and release the lock."""
        try:
            await asyncio.sleep(random.random())  # simulate work with random duration
            async with lock:
                print(f"Task {task_id} acquired lock")
                await asyncio.sleep(random.random())  # simulate work with random duration
        except asyncio.CancelledError:
            print(f"Task {task_id} was cancelled")
            cancellation_occurred.set()

    async def monitor_func():
        """Function to be run in monitor task. Randomly cancels one of the tasks."""
        await asyncio.sleep(random.random())  # wait random duration before cancelling a task
        order = list(range(len(tasks)))
        random.shuffle(order)
        for i in order:
            task_to_cancel = tasks[i]
            if not task_to_cancel.done():
                task_to_cancel.cancel()

    # Create tasks
    for i in range(num_tasks):
        tasks.append(asyncio.create_task(task_func(i)))

    await asyncio.sleep(0)
    # Create monitor task
    monitor_task = asyncio.create_task(monitor_func())

    # Wait for all tasks to complete or be cancelled
    await asyncio.gather(*tasks, return_exceptions=True)

    await monitor_task

    # At least one cancellation should have occurred
    assert cancellation_occurred.is_set()


class DelayedFairAsyncRLock(FairAsyncRLock):
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Release the lock"""
        await asyncio.sleep(0.1)
        await super().__aexit__(exc_type, exc_val, exc_tb)


@pytest.mark.asyncio
async def test_delayed_release():
    lock = DelayedFairAsyncRLock()

    async def first_task():
        async with lock:
            await asyncio.sleep(0.1)  # hold lock for a moment

    async def second_task():
        await asyncio.sleep(0.05)  # wait until first task has lock
        await lock.acquire()
        got_lock = lock.is_owner()
        lock.release()
        return got_lock

    t1 = asyncio.create_task(first_task())
    t2 = asyncio.create_task(second_task())
    await asyncio.gather(t1, t2)

    assert t2.result() is True, "Second task should acquire the lock after the first task has released it"


class ExceptionFairAsyncRLock(FairAsyncRLock):
    def release(self):
        """Release the lock"""
        raise asyncio.CancelledError()
        super().release()


@pytest.mark.asyncio
async def _test_exception_on_release_gh7():
    lock = ExceptionFairAsyncRLock()

    async def task():
        with pytest.raises(asyncio.CancelledError):
            async with lock:
                pass  # no action needed inside

    await asyncio.create_task(task())
    assert lock._owner is None, "Lock owner should be None after an exception"
    assert len(lock._queue) == 0, "Lock queue should be empty after an exception"


@pytest.mark.asyncio
async def test_fair_async_rlock_deadlock_scenario_regression_gh14():
    lock = FairAsyncRLock()

    # Use events to control the order of execution
    task1_aquired = asyncio.Event()
    task2_started = asyncio.Event()
    task3_started = asyncio.Event()
    task3_acquired = asyncio.Event()
    task4_acquired = asyncio.Event()
    task4_started = asyncio.Event()
    task1_done = asyncio.Event()

    async def task1():
        async with lock:
            task1_aquired.set()
            await task2_started.wait()
            await task3_started.wait()  # wait until Task 3 in queue too
            await task4_started.wait()  # wait until Task 4 in queue too
            await asyncio.sleep(0.1)  # make sure Task 4 gets in queue
            task1_done.set()  # signal done before release, so Task 2 can cancel Task 3

    async def task2():
        await task1_aquired.wait()
        task2_started.set()
        await task1_done.wait()  # Wait until Task 1 done then cancel Task 3
        t3.cancel()

    async def task3():
        await task2_started.wait()
        task3_started.set()
        async with lock:  # now in queue, waiting
            task3_acquired.set()  # Should not get reached, because Task 3 will be cancelled always

    async def task4():
        await task3_started.wait()
        await asyncio.sleep(0.1)  # make sure Task 3 gets in queue first
        task4_started.set()
        async with lock:  # it's now in queue, just after Task 3, and waiting
            task4_acquired.set()  # Will get set if no bug

    t1 = asyncio.create_task(task1())
    t2 = asyncio.create_task(task2())
    t3 = asyncio.create_task(task3())
    t4 = asyncio.create_task(task4())

    await t1
    await t2
    with pytest.raises(asyncio.CancelledError):
        await t3  # Here we get ValueError: <asyncio.locks.Event object at 0x7f419f7dba90 [set]> is not in deque
    # Task 3 would never acquire
    assert not task3_acquired.is_set()

    # Task 4 should not deadlock. It should be able to acquire the locks
    await asyncio.wait([t4], timeout=1)
    assert task4_acquired.is_set()


@pytest.mark.asyncio
async def test_gh17_regression():
    lock = FairAsyncRLock()

    # Use events to control the order of execution
    task1_aquired = asyncio.Event()
    # task1_released = asyncio.Event()
    task2_acquired = asyncio.Event()

    async def task1():
        async with lock:
            # tell task 2 to acquire lock
            task1_aquired.set()
            # but sleep long enough to make sure task 2 waits on lock before we release this one.
            await asyncio.sleep(0.1)
        # task1_released.set()

    async def task2():
        await task1_aquired.wait()
        async with lock:
            task2_acquired.set()
            # hold this for long enough to allow task3 to release first in a race condition
            # in which task3 beats task2 to ownership and then releases only to find
            # task2 has set ownership, giving a foreign lock assert scenario
            await asyncio.sleep(0.2)

    async def task3():
        # NB: we achieve this race condition with sleep(0.1)
        # awaiting task1_released does not give it the edge
        await asyncio.sleep(0.1)
        # await task1_released.wait()
        async with lock:
            # if task 3 were to get an immediate lock (beating task3 to ownership), then waiting
            # for this means we don't release until just after task2
            # has clobbered over our ownership.
            await task2_acquired.wait()

    t1 = asyncio.create_task(task1())
    t2 = asyncio.create_task(task2())
    t3 = asyncio.create_task(task3())

    await asyncio.gather(t1, t2, t3)
