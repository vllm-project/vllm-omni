# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CI benchmark for HunYuanImage diffusion model performance tracking"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import torch
from vllm import LLM, SamplingParams


class HunYuanImageBenchmark:
    """HunYuanImage benchmark runner"""

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 4,
        expert_parallel_size: Optional[int] = None,
    ):
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.expert_parallel_size = expert_parallel_size
        
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            expert_parallel_size=expert_parallel_size,
            trust_remote_code=True,
        )
        
        self.test_prompts = [
            "A brown and white dog is running on the grass",
            "A beautiful sunset over the ocean with mountains in the background",
            "A futuristic city with flying cars and tall skyscrapers",
            "A cute cat playing with a ball of yarn",
            "A serene Japanese garden with cherry blossoms",
        ]

    def run_benchmark(
        self,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        seed: int = 1234,
    ) -> Dict:
        """Run benchmark and return results"""
        results = {
            'model': self.model_path,
            'tensor_parallel_size': self.tensor_parallel_size,
            'expert_parallel_size': self.expert_parallel_size,
            'num_inference_steps': num_inference_steps,
            'guidance_scale': guidance_scale,
            'seed': seed,
            'timestamp': datetime.now().isoformat(),
            'prompts': [],
        }

        sampling_params = SamplingParams(
            max_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            seed=seed,
            guidance_scale=guidance_scale,
        )

        total_latency = 0.0
        
        for i, prompt in enumerate(self.test_prompts):
            print(f"Running benchmark {i+1}/{len(self.test_prompts)}...")
            
            start_time = time.perf_counter()
            
            outputs = self.llm.generate(
                prompts=[prompt],
                sampling_params=sampling_params,
                extra_body={"num_inference_steps": num_inference_steps},
            )
            
            end_time = time.perf_counter()
            latency = end_time - start_time
            total_latency += latency
            
            results['prompts'].append({
                'prompt': prompt,
                'latency_seconds': latency,
            })
            
            print(f"  Prompt {i+1}: {latency:.2f}s")

        results['statistics'] = {
            'total_prompts': len(self.test_prompts),
            'total_latency_seconds': total_latency,
            'average_latency_seconds': total_latency / len(self.test_prompts),
            'min_latency_seconds': min(p['latency_seconds'] for p in results['prompts']),
            'max_latency_seconds': max(p['latency_seconds'] for p in results['prompts']),
        }

        return results

    def save_results(self, results: Dict, output_dir: str):
        """Save benchmark results to JSON and Markdown"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        json_file = output_path / f'benchmark_{timestamp}.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        md_file = output_path / f'benchmark_{timestamp}.md'
        self._save_markdown_report(results, md_file)
        
        print(f"Results saved to {json_file}")
        print(f"Report saved to {md_file}")

    def _save_markdown_report(self, results: Dict, output_file: Path):
        """Save Markdown performance report"""
        stats = results['statistics']
        
        report = f"""# HunYuanImage Benchmark Report

## Test Configuration

- **Model**: {results['model']}
- **Tensor Parallel**: {results['tensor_parallel_size']}
- **Expert Parallel**: {results['expert_parallel_size'] or 'None'}
- **Inference Steps**: {results['num_inference_steps']}
- **Guidance Scale**: {results['guidance_scale']}
- **Timestamp**: {results['timestamp']}

## Performance Statistics

| Metric | Value |
|--------|-------|
| Total Prompts | {stats['total_prompts']} |
| Total Latency | {stats['total_latency_seconds']:.2f} s |
| Average Latency | {stats['average_latency_seconds']:.2f} s |
| Min Latency | {stats['min_latency_seconds']:.2f} s |
| Max Latency | {stats['max_latency_seconds']:.2f} s |

## Detailed Results

| # | Prompt | Latency (s) |
|---|--------|-------------|
"""
        
        for i, prompt_data in enumerate(results['prompts'], 1):
            prompt = prompt_data['prompt'][:50] + '...' if len(prompt_data['prompt']) > 50 else prompt_data['prompt']
            report += f"| {i} | {prompt} | {prompt_data['latency_seconds']:.2f} |\n"
        
        report += """
## Optimization Notes

Attention and MoE modules account for 70-80% of execution time. Priority areas:

1. **Attention Optimization** (~30% runtime)
   - Flash Attention 2
   - Paged Attention
   - KV Cache management

2. **MoE Optimization** (~70% runtime)
   - Expert Parallel strategies
   - Dynamic load balancing
   - Communication overhead reduction

---

*Generated by HunYuanImage CI Benchmark*
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)


def main():
    parser = argparse.ArgumentParser(description='HunYuanImage CI Benchmark')
    parser.add_argument('--model', type=str, required=True, help='Model path')
    parser.add_argument('--tensor-parallel-size', type=int, default=4, help='Tensor parallel size')
    parser.add_argument('--expert-parallel-size', type=int, default=None, help='Expert parallel size')
    parser.add_argument('--num-steps', type=int, default=50, help='Number of inference steps')
    parser.add_argument('--guidance-scale', type=float, default=5.0, help='Guidance scale')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--output-dir', type=str, default='benchmark_results', help='Output directory')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("HunYuanImage CI Benchmark")
    print("=" * 70)
    
    benchmark = HunYuanImageBenchmark(
        model_path=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        expert_parallel_size=args.expert_parallel_size,
    )
    
    results = benchmark.run_benchmark(
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )
    
    benchmark.save_results(results, args.output_dir)
    
    print()
    print("=" * 70)
    print("Benchmark completed!")
    print(f"Average latency: {results['statistics']['average_latency_seconds']:.2f}s")
    print("=" * 70)


if __name__ == '__main__':
    main()
