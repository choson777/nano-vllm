import asyncio
import aiohttp
import time
import json

async def my_coroutine(id:int,results:list):
    print(f"Coroutine {id} is executing...")
    await asyncio.sleep(id)
    print(f"Coroutine {id} is done.")
    results.append(f"result of Coroutine {id}")

async def main():
    results = []
    tasks = []
    task = asyncio.create_task(my_coroutine(10,results))
    tasks.append(task)
    await asyncio.sleep(5)
    task = asyncio.create_task(my_coroutine(20,results))
    tasks.append(task)
    print(results)
    # await asyncio.sleep(12)
    await asyncio.gather(*tasks)
    print(results)

asyncio.run(main())


