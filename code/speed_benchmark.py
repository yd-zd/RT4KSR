# code/speed_benchmark.py
import os, torch, argparse, pathlib, time
import numpy as np
from utils import image
import model
from test import load_checkpoint, reparameterize

class Config:
    """Config object that matches the structure from parser.base_parser()"""
    def __init__(self, scale=2, checkpoint_id="rt4ksr_x2", use_rep=True):
        self.seed = 1
        self.dataroot = os.path.join(pathlib.Path.home(), "datasets/image_restoration")
        self.benchmark = ["ntire23rtsr"]
        self.checkpoints_root = "code/checkpoints"
        self.checkpoint_id = checkpoint_id

        # model definitions
        self.bicubic = False
        self.arch = "rt4ksr_rep"
        self.feature_channels = 24
        self.num_blocks = 4
        self.act_type = "gelu"
        self.is_train = True  # Must be True first to load training weights
        self.rep = use_rep
        self.save_rep_checkpoint = False

        # data
        self.scale = scale
        self.rgb_range = 1.0

def setup_model(scale, ckpt_id, use_rep=True):
    """Setup and return the model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----- build network using original author's approach -----
    config = Config(scale=scale, checkpoint_id=ckpt_id, use_rep=use_rep)

    # Use original author's dynamic model loading
    net = torch.nn.DataParallel(
        model.__dict__[config.arch](config)
    ).to(device)
    net = load_checkpoint(net, device, config.checkpoint_id)

    # Apply reparameterization if requested (like original test.py)
    if config.rep:
        net = reparameterize(config, net, device)

    net.eval()
    return net, device

def benchmark_inference(net, device, lr_uint, num_iterations=100, warmup_iterations=10, timing_mode='gpu-only'):
    """
    Benchmark inference speed over multiple iterations simulating streaming video

    Args:
        net: The neural network model
        device: The device (CPU/GPU) the model is on
        lr_uint: Raw input image data (HWC uint8 RGB) - will be processed fresh each iteration
        num_iterations: Number of inference iterations to measure
        warmup_iterations: Number of warmup iterations (not measured)
        timing_mode: 'gpu-only' (default) measures only GPU inference, 'full-pipeline' includes CPU preprocessing

    Returns:
        dict: Dictionary containing timing statistics (min, avg, max in milliseconds)
    """
    if timing_mode == 'gpu-only':
        print(f"Running benchmark with {num_iterations} iterations (+{warmup_iterations} warmup iterations)...")
        print("Timing Mode: GPU-only (excludes CPU preprocessing from measurements)")
    else:
        print(f"Running benchmark with {num_iterations} iterations (+{warmup_iterations} warmup iterations)...")
        print("Timing Mode: Full pipeline (includes CPU preprocessing in measurements)")

    # Optimized preprocessing strategy based on timing mode
    if timing_mode == 'gpu-only':
        # GPU-only mode: preprocess once, reuse tensor for all iterations
        print("Preprocessing tensor once for all iterations...")
        lr_tensor = image.uint2tensor4(lr_uint).to(device)

        # Warmup iterations (not measured) - reuse preprocessed tensor
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = net(lr_tensor)

        # Actual benchmark iterations - reuse preprocessed tensor, time only GPU inference
        inference_times = []
        with torch.no_grad():
            for i in range(num_iterations):
                if device.type == 'cuda':
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    _ = net(lr_tensor)  # Same tensor, only GPU inference timed
                    end_event.record()
                    torch.cuda.synchronize()
                    inference_time_ms = start_event.elapsed_time(end_event)
                else:
                    start_time = time.time()
                    _ = net(lr_tensor)
                    end_time = time.time()
                    inference_time_ms = (end_time - start_time) * 1000

                inference_times.append(inference_time_ms)

                if (i + 1) % 10 == 0:  # Progress indicator every 10 iterations
                    print(f"  Completed {i + 1}/{num_iterations} iterations...")
    else:
        # Full pipeline mode: fresh preprocessing each iteration
        print("Fresh preprocessing each iteration to simulate streaming...")

        # Warmup iterations (not measured) - each processes the data fresh
        with torch.no_grad():
            for _ in range(warmup_iterations):
                lr_tensor = image.uint2tensor4(lr_uint).to(device)
                _ = net(lr_tensor)

        # Actual benchmark iterations - each processes the data fresh
        inference_times = []
        with torch.no_grad():
            for i in range(num_iterations):
                if device.type == 'cuda':
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    lr_tensor = image.uint2tensor4(lr_uint).to(device)  # Fresh preprocessing
                    _ = net(lr_tensor)
                    end_event.record()
                    torch.cuda.synchronize()
                    inference_time_ms = start_event.elapsed_time(end_event)
                else:
                    start_time = time.time()
                    lr_tensor = image.uint2tensor4(lr_uint).to(device)
                    _ = net(lr_tensor)
                    end_time = time.time()
                    inference_time_ms = (end_time - start_time) * 1000

                inference_times.append(inference_time_ms)

                if (i + 1) % 10 == 0:  # Progress indicator every 10 iterations
                    print(f"  Completed {i + 1}/{num_iterations} iterations...")

    # Calculate statistics
    min_time = min(inference_times)
    avg_time = sum(inference_times) / len(inference_times)
    max_time = max(inference_times)

    return {
        'min_ms': min_time,
        'avg_ms': avg_time,
        'max_ms': max_time,
        'all_times': inference_times
    }

def run_speed_benchmark(lr_path, scale, ckpt_id, use_rep=True, num_iterations=100, warmup_iterations=10):
    """Run complete speed benchmark for an image"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Setup model
    print("Setting up model...")
    net, device = setup_model(scale, ckpt_id, use_rep)

    # Read input image once (file I/O only happens here, not during timing)
    print(f"Loading input image: {lr_path}")
    lr_uint = image.imread_uint(lr_path, n_channels=3)           # HWC uint8 RGB

    print(f"Input image shape: {lr_uint.shape}")
    print(f"Scale factor: {scale}x")
    print(f"Reparameterization: {'Enabled' if use_rep else 'Disabled'}")

    if args.timing_mode == 'gpu-only':
        print("Mode: GPU-only - preprocessing once, timing only GPU inference")
    else:
        print("Mode: Full pipeline - fresh preprocessing each iteration")

    # Run benchmark with optimized preprocessing strategy
    stats = benchmark_inference(net, device, lr_uint, num_iterations, warmup_iterations, args.timing_mode)

    # Print results
    print("\n" + "="*50)
    print("SPEED BENCHMARK RESULTS")
    print("="*50)
    print(f"Min inference time:  {stats['min_ms']:.2f} ms")
    print(f"Avg inference time:  {stats['avg_ms']:.2f} ms")
    print(f"Max inference time:  {stats['max_ms']:.2f} ms")
    print(f"Total iterations:    {len(stats['all_times'])}")

    # Calculate FPS
    avg_fps = 1000.0 / stats['avg_ms']  # Convert ms to FPS
    print(f"Average FPS:         {avg_fps:.2f} FPS")

    # Performance analysis
    times_array = np.array(stats['all_times'])
    std_dev = np.std(times_array)
    median_time = np.median(times_array)

    print(f"Std Deviation:       {std_dev:.2f} ms")
    print(f"Median time:         {median_time:.2f} ms")

    # Variability check (coefficient of variation)
    if stats['avg_ms'] > 0:
        cv = (std_dev / stats['avg_ms']) * 100
        print(f"Variability (CV%):   {cv:.1f}%")
        if cv < 3:
            print("  → Very low variability: Extremely stable performance")
        elif cv < 7:
            print("  → Low variability: Stable performance")
        elif cv < 15:
            print("  → Moderate variability: Acceptable performance consistency")
        else:
            print("  → High variability: Check for external factors or system instability")

    # Memory info if CUDA is available
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(device) / 1024 / 1024  # MB
        memory_reserved = torch.cuda.memory_reserved(device) / 1024 / 1024    # MB
        print(f"GPU memory allocated: {memory_allocated:.2f} MB")
        print(f"GPU memory reserved:  {memory_reserved:.2f} MB")

    print("="*50)

    return stats

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Speed benchmark for RT4KSR model")
    p.add_argument('--input', required=True, help='path to low-res image')
    p.add_argument('--scale', type=int, default=2, choices=[2,3], help='upscaling scale factor')
    p.add_argument('--iterations', type=int, default=100, help='number of benchmark iterations')
    p.add_argument('--warmup', type=int, default=10, help='number of warmup iterations (not measured)')
    p.add_argument('--no-rep', action='store_true', help='disable reparameterization (slower but matches training)')
    p.add_argument('--runs', type=int, default=1, help='number of benchmark runs for consistency check')
    p.add_argument('--detailed', action='store_true', help='enable detailed performance analysis')
    p.add_argument('--timing-mode', choices=['gpu-only', 'full-pipeline'], default='gpu-only',
                   help='gpu-only: measure only GPU inference time (default), full-pipeline: include CPU preprocessing')
    p.add_argument('--cooling-time', type=float, default=0, help='seconds to wait between runs for GPU cooling (default: 0)')
    args = p.parse_args()

    ckpt = f"rt4ksr_x{args.scale}"

    # Run multiple benchmarks for consistency check
    all_runs_stats = []

    for run in range(args.runs):
        if args.runs > 1:
            print(f"\n--- Benchmark Run {run + 1}/{args.runs} ---")

        stats = run_speed_benchmark(
            lr_path=args.input,
            scale=args.scale,
            ckpt_id=ckpt,
            use_rep=not args.no_rep,
            num_iterations=args.iterations,
            warmup_iterations=args.warmup
        )
        all_runs_stats.append(stats)

        # Add cooling time between runs (except after the last run)
        if args.cooling_time > 0 and run < args.runs - 1:
            print(f"Waiting {args.cooling_time}s for GPU cooling...")
            time.sleep(args.cooling_time)

    # Multi-run analysis
    if args.runs > 1:
        print(f"\n{'='*50}")
        print("MULTI-RUN CONSISTENCY ANALYSIS")
        print(f"{'='*50}")

        avg_times_across_runs = [run_stats['avg_ms'] for run_stats in all_runs_stats]
        fps_across_runs = [1000.0 / avg_time for avg_time in avg_times_across_runs]

        print(f"Run count:           {args.runs}")
        print(f"Avg times across runs: {avg_times_across_runs}")
        print(f"FPS across runs:     {fps_across_runs}")
        print(f"Mean FPS:            {np.mean(fps_across_runs):.2f} FPS")
        print(f"FPS Std Dev:         {np.std(fps_across_runs):.2f} FPS")
        print(f"FPS Range:           {min(fps_across_runs):.2f} - {max(fps_across_runs):.2f} FPS")

        # Consistency check with thermal effect consideration
        fps_cv = (np.std(fps_across_runs) / np.mean(fps_across_runs)) * 100 if np.mean(fps_across_runs) > 0 else 0
        print(f"FPS Consistency (CV%): {fps_cv:.1f}%")

        # Check for thermal degradation pattern (performance decreasing over runs)
        if len(fps_across_runs) >= 3:
            first_half = np.mean(fps_across_runs[:len(fps_across_runs)//2])
            second_half = np.mean(fps_across_runs[len(fps_across_runs)//2:])
            thermal_drop = (first_half - second_half) / first_half * 100
            if thermal_drop > 3:  # More than 3% performance drop
                print(f"Thermal degradation:   {thermal_drop:.1f}% performance drop detected")
                print("  → Consider cooling or reducing benchmark duration")

        if fps_cv < 2:
            print("  → Excellent consistency: Results are highly reliable")
        elif fps_cv < 5:
            print("  → Good consistency: Results are reliable")
        elif fps_cv < 10:
            print("  → Moderate consistency: Results are acceptable with minor variability")
            if fps_cv > 6:
                print("  → Note: Higher variability may indicate thermal effects")
        else:
            print("  → Poor consistency: Check for system variability or thermal effects")
            print("  → Recommendation: Allow GPU cooling time between runs or reduce benchmark duration")

        print(f"{'='*50}")
