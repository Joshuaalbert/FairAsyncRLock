from fair_async_rlock.asyncio_fair_async_rlock import *

try:
    from fair_async_rlock.anyio_fair_async_rlock import *
except ImportError:
    pass
