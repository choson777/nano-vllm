import asyncio
import aiohttp
import time

async def make_request(session, url, prompt, request_id):
    """发起请求并在完成后立即处理"""
    payload = {"prompt": prompt, "max_tokens": 64}
    print(f"[{time.strftime('%H:%M:%S')}] 🚀 请求 {request_id} 已发起: '{prompt}'")
    
    async with session.post(url, json=payload) as response:
        result = await response.json()
        print(f"[{time.strftime('%H:%M:%S')}] ✅ 请求 {request_id} 完成: '{prompt[:10]}...'")
        return request_id, result

async def coordinator(url, prompts):
    """协调多个请求，控制发起间隔"""
    async with aiohttp.ClientSession() as session:
        tasks = []
        start_time = time.time()
        
        for i, prompt in enumerate(prompts):
            # 1. 发起请求（立即执行）
            task = asyncio.create_task(
                make_request(session, url, prompt, i+1)
            )
            tasks.append(task)
            
            # 2. 打印当前状态
            elapsed = time.time() - start_time
            print(f"[{time.strftime('%H:%M:%S')}] ⏱️  已运行 {elapsed:.2f}秒，当前活跃请求: {len(tasks)}")
            
            # 3. 如果不是最后一个请求，等待0.5秒再发起下一个
            if i < len(prompts) - 1:
                await asyncio.sleep(0.1)
                print(f"[{time.strftime('%H:%M:%S')}] ⏸️  等待0.5秒后发起请求 {i+2}")
        
        # 4. 按完成顺序处理结果（而非发起顺序）
        for future in asyncio.as_completed(tasks):
            request_id, result = await future
            print(f"[{time.strftime('%H:%M:%S')}][!] 请求 {request_id} 异步返回结果")

async def main():
    url = "http://localhost:8000/generate"
    prompts = [
        "快速计算: 15 × 24 = ?" * 3,  # 长提示
        "简短问题: 2+2=?",             # 短提示
        "中等问题: 解释牛顿第一定律",   # 中等长度
        # "超长提示: " + "测试 " * 100   # 非常长的提示
    ]
    
    print("=== 开始异步请求测试 ===")
    await coordinator(url, prompts)
    print("=== 所有请求处理完成 ===")

if __name__ == "__main__":
    asyncio.run(main())