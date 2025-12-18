import time
import argparse
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

# Try to import pynvml for GPU monitoring
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("Warning: 'nvidia-ml-py' not found. Memory monitoring will be disabled.")

# ================= 配置部分 =================
API_KEY = "EMPTY" 
API_BASE = "http://localhost:8000/v1"
MODEL_NAME = "your-model-name" # 请修改为你的模型名称

# ================= 显存监控类 =================
class GPUMonitor(threading.Thread):
    def __init__(self, interval=0.1):
        super().__init__()
        self.interval = interval
        self.running = True
        self.max_memory_used = 0  # MB
        self.device_count = 0
        self.error = False

    def run(self):
        if not PYNVML_AVAILABLE:
            return
        
        try:
            pynvml.nvmlInit()
            self.device_count = pynvml.nvmlDeviceGetCount()
            print(f"[Monitor] Monitoring {self.device_count} GPU(s)...")
            
            while self.running:
                current_total_mem = 0
                # 获取所有GPU显存之和，或者你可以修改为只监控指定GPU
                for i in range(self.device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    current_total_mem += info.used
                
                # 转换为 MB
                current_total_mem_mb = current_total_mem / 1024 / 1024
                if current_total_mem_mb > self.max_memory_used:
                    self.max_memory_used = current_total_mem_mb
                
                time.sleep(self.interval)
                
        except Exception as e:
            print(f"[Monitor] Error reading GPU stats: {e}")
            self.error = True
        finally:
            if PYNVML_AVAILABLE:
                try:
                    pynvml.nvmlShutdown()
                except:
                    pass

    def stop(self):
        self.running = False
        self.join()

# ================= 请求发送函数 =================
def send_request(client, prompt, images_per_req):
    start_time = time.time()
    try:
        # 发送请求
        client.images.generate(
            model=MODEL_NAME,
            prompt=prompt,
            n=images_per_req, # 这里模拟 Batch Size (如果服务端支持)
            size="1024x1024", 
            response_format="b64_json" # 只拿数据不下载图片，减少网络干扰
        )
        latency = time.time() - start_time
        return latency, True
    except Exception as e:
        print(f"Request failed: {e}")
        return time.time() - start_time, False

# ================= 主测试逻辑 =================
def benchmark(concurrency, num_requests, prompts, images_per_req=1):
    client = OpenAI(api_key=API_KEY, base_url=API_BASE)
    
    print(f"--- Benchmark Config ---")
    print(f"Concurrency (Clients): {concurrency}")
    print(f"Total Requests: {num_requests}")
    print(f"Images per Request (Batch): {images_per_req}")
    print(f"Expected Total Images: {num_requests * images_per_req}")
    
    # 1. 启动显存监控
    gpu_monitor = GPUMonitor()
    if PYNVML_AVAILABLE:
        gpu_monitor.start()
    else:
        print("[Info] Skipping memory check (run on GPU server to enable).")

    # 2. 预热 (Warmup) - 可选，避免第一次编译图影响结果
    print("\nWarming up...")
    try:
        send_request(client, prompts[0], 1)
    except:
        pass

    # 3. 开始压测
    print("\nStarting Benchmark...")
    latencies = []
    success_requests = 0
    
    start_benchmark = time.time()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(send_request, client, prompts[i % len(prompts)], images_per_req) 
            for i in range(num_requests)
        ]
        
        for future in futures:
            lat, success = future.result()
            latencies.append(lat)
            if success:
                success_requests += 1

    total_time = time.time() - start_benchmark
    
    # 4. 停止监控
    if PYNVML_AVAILABLE:
        gpu_monitor.stop()

    # ================= 结果计算 =================
    # 核心指标 1: Throughput (Images / Second)
    # 公式: (成功的请求数 * 单次请求图片数) / 总时间
    total_images_generated = success_requests * images_per_req
    throughput_img_s = total_images_generated / total_time
    
    # 核心指标 2: Memory Peak
    memory_peak_str = f"{gpu_monitor.max_memory_used:.2f} MB" if PYNVML_AVAILABLE and not gpu_monitor.error else "N/A (Not available)"

    latencies = np.array(latencies)
    avg_latency = np.mean(latencies)
    p99 = np.percentile(latencies, 99)

    print("\n" + "="*40)
    print("       BENCHMARK REPORT       ")
    print("="*40)
    print(f"Total Time Taken:       {total_time:.2f} s")
    print(f"Total Images Gen:       {total_images_generated}")
    print(f"Success Rate:           {success_requests/num_requests*100:.1f}%")
    print("-" * 40)
    print(f"1. Throughput:          {throughput_img_s:.2f} images/s")
    print(f"2. Memory Peak (Total): {memory_peak_str}")
    print("-" * 40)
    print(f"Avg Latency:            {avg_latency:.2f} s")
    print(f"P99 Latency:            {p99:.2f} s")
    print("="*40)

if __name__ == "__main__":
    # 构造一些不同长度的 Prompt
    dummy_prompts = [
        "A cat",
        "A futuristic city with flying cars and neon lights, cyberpunk style, high resolution, 8k",
        "An astronaut riding a horse on mars, realistic photography",
        "A landscape painting of mountains and river, oil painting style"
    ]
    
    # 运行配置
    # concurrency: 并发数 (模拟多少个用户同时在发请求)
    # num_requests: 总请求数
    # images_per_req: 单个请求生成的图片数 (OpenAI API的 'n' 参数)
    
    # 场景：测试极限吞吐量
    # 建议：逐渐调大 concurrency，观察 Throughput 何时不再上升，或者观察 Memory Peak 何时接近显存上限
    benchmark(concurrency=8, num_requests=40, prompts=dummy_prompts, images_per_req=1)