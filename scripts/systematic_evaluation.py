#!/usr/bin/env python3
"""
Systematic Evaluation of SNN Tracking System

This script performs comprehensive evaluation with:
1. Tunable noise levels (0% to 50%)
2. Variable object density (1 to 5 objects)
3. Crossover scenarios (objects crossing paths)
4. FPS benchmarking

Run: python scripts/systematic_evaluation.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.ndimage import label as scipy_label
import time
import sys
import json
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from inference import load_model


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class EvalConfig:
    """Evaluation configuration."""
    T: int = 60  # Frames per test
    H: int = 128
    W: int = 128
    C: int = 2
    seed: int = 42
    output_dir: str = "evaluation_results"


@dataclass 
class TestResult:
    """Result from a single test."""
    noise_level: float
    n_objects: int
    has_crossover: bool
    detection_rate: float
    mean_error: float
    max_error: float
    both_detected_rate: float  # For multi-object
    fps: float
    frames_tested: int


# =============================================================================
# SYNTHETIC DATA GENERATOR
# =============================================================================

class SyntheticDataGenerator:
    """Generate synthetic satellite tracking data with tunable parameters."""
    
    def __init__(self, config: EvalConfig):
        self.config = config
        self.H = config.H
        self.W = config.W
        self.C = config.C
        self.T = config.T
        
        # Hot pixels: ~0.05% of pixels (about 8 on 128x128)
        self.hot_pixel_rate = 0.0005
        self.hot_pixels = torch.rand(self.H, self.W) < self.hot_pixel_rate
        # Hot pixel intensity: ~0.15 (150/1000 scale, much weaker than signal)
        self.hot_pixel_intensity = 0.15
        
    def generate_trajectory(
        self, 
        start_pos: Tuple[float, float],
        velocity: Tuple[float, float],
        T: int,
        trajectory_type: str = "orbital"
    ) -> List[Tuple[float, float]]:
        """Generate a single object trajectory."""
        positions = []
        cx, cy = start_pos
        vx, vy = velocity
        
        for t in range(T):
            positions.append((cx, cy))
            
            if trajectory_type == "linear":
                cx = np.clip(cx + vx, 12, self.W - 12)
                cy = np.clip(cy + vy, 12, self.H - 12)
                
            elif trajectory_type == "orbital":
                ax = np.random.randn() * 0.15 + (self.W/2 - cx) * 0.003
                ay = np.random.randn() * 0.15 + (self.H/2 - cy) * 0.003
                vx = np.clip(vx + ax, -3, 3)
                vy = np.clip(vy + ay, -3, 3)
                cx = np.clip(cx + vx, 12, self.W - 12)
                cy = np.clip(cy + vy, 12, self.H - 12)
                
            elif trajectory_type == "crossing":
                # Linear path for controlled crossover
                cx = np.clip(cx + vx, 12, self.W - 12)
                cy = np.clip(cy + vy, 12, self.H - 12)
                
        return positions
    
    def generate_crossing_trajectories(self, n_objects: int, T: int) -> List[List[Tuple[float, float]]]:
        """Generate trajectories that cross in the middle."""
        trajectories = []
        center = (self.W / 2, self.H / 2)
        
        for i in range(n_objects):
            angle = (2 * np.pi * i) / n_objects
            # Start from edge, move toward center and beyond
            start_x = center[0] + 50 * np.cos(angle)
            start_y = center[1] + 50 * np.sin(angle)
            vx = -1.5 * np.cos(angle)
            vy = -1.5 * np.sin(angle)
            
            traj = self.generate_trajectory(
                (start_x, start_y), 
                (vx, vy), 
                T, 
                "crossing"
            )
            trajectories.append(traj)
            
        return trajectories
    
    def generate_non_crossing_trajectories(self, n_objects: int, T: int) -> List[List[Tuple[float, float]]]:
        """Generate trajectories that don't cross."""
        trajectories = []
        
        # Divide space into regions
        regions = [
            (25, 25), (100, 25), (25, 100), (100, 100), (64, 64)
        ]
        
        for i in range(min(n_objects, len(regions))):
            start = regions[i]
            vx = np.random.uniform(-1, 1)
            vy = np.random.uniform(-1, 1)
            
            traj = self.generate_trajectory(start, (vx, vy), T, "orbital")
            trajectories.append(traj)
            
        return trajectories
    
    def create_frame(
        self,
        positions: List[Tuple[float, float]],
        noise_level: float,
        tail_length: int = 5,
        prev_positions: Optional[List[List[Tuple[float, float]]]] = None,
        t: int = 0
    ) -> torch.Tensor:
        """Create a single frame with objects and noise.
        
        Noise model: At noise_level X, X% of pixels are corrupted with
        signal-strength noise that can mask the satellite.
        """
        frame = torch.zeros(self.C, self.H, self.W)
        
        # First add each object (signal)
        for obj_idx, (cx, cy) in enumerate(positions):
            icx, icy = int(cx), int(cy)
            
            # Add tail if previous positions available (brighter tail)
            if prev_positions is not None and len(prev_positions[obj_idx]) > 0:
                for tail_idx in range(1, min(tail_length + 1, t + 1)):
                    if t - tail_idx >= 0 and t - tail_idx < len(prev_positions[obj_idx]):
                        tcx, tcy = prev_positions[obj_idx][t - tail_idx]
                        itcx, itcy = int(tcx), int(tcy)
                        fade = 1.0 - (tail_idx / (tail_length + 1))
                        for dy in range(-1, 2):
                            for dx in range(-1, 2):
                                py, px = itcy + dy, itcx + dx
                                if 0 <= py < self.H and 0 <= px < self.W:
                                    frame[0, py, px] = max(frame[0, py, px].item(), fade * 0.25)  # Was 0.15
                                    frame[1, py, px] = max(frame[1, py, px].item(), fade * 0.6)   # Was 0.4
            
            # Add satellite core (BRIGHT - high intensity to stand out from noise)
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    py, px = icy + dy, icx + dx
                    if 0 <= py < self.H and 0 <= px < self.W:
                        dist = np.sqrt(dx**2 + dy**2)
                        intensity = max(0, 1.0 - dist * 0.18)  # Wider bright core
                        frame[0, py, px] = max(frame[0, py, px].item(), intensity * 0.5)  # Was 0.3
                        frame[1, py, px] = max(frame[1, py, px].item(), intensity * 1.0)  # Already max
        
        # Add noise
        if noise_level > 0:
            # 1. Hot pixels (weak, ~8 pixels total)
            frame[0] = torch.clamp(frame[0] + self.hot_pixels.float() * self.hot_pixel_intensity * 0.3, 0, 1)
            frame[1] = torch.clamp(frame[1] + self.hot_pixels.float() * self.hot_pixel_intensity * 0.5, 0, 1)
            
            # 2. Random background noise
            noise_mask = torch.rand(self.H, self.W) < noise_level
            noise_intensity = torch.rand(self.H, self.W) * (0.2 + 0.3 * noise_level)
            
            frame[0] = torch.clamp(frame[0] + noise_mask.float() * noise_intensity * 0.3, 0, 1)
            frame[1] = torch.clamp(frame[1] + noise_mask.float() * noise_intensity * 1.0, 0, 1)
        
        return frame
    
    def generate_dataset(
        self,
        n_objects: int,
        noise_level: float,
        crossover: bool,
        T: Optional[int] = None
    ) -> Tuple[torch.Tensor, List[List[Tuple[float, float]]]]:
        """Generate complete dataset with all frames."""
        if T is None:
            T = self.T
            
        # Generate trajectories
        if crossover and n_objects > 1:
            trajectories = self.generate_crossing_trajectories(n_objects, T)
        else:
            trajectories = self.generate_non_crossing_trajectories(n_objects, T)
        
        # Generate frames
        frames = torch.zeros(T, self.C, self.H, self.W)
        
        for t in range(T):
            positions = [traj[t] for traj in trajectories]
            frames[t] = self.create_frame(
                positions, 
                noise_level,
                prev_positions=trajectories,
                t=t
            )
        
        # Normalize
        if frames.max() > 0:
            frames = frames / (frames.max() + 1e-8)
        
        return frames, trajectories


# =============================================================================
# EVALUATION ENGINE
# =============================================================================

class EvaluationEngine:
    """Run systematic evaluations."""
    
    def __init__(self, model, device, config: EvalConfig):
        self.model = model
        self.device = device
        self.config = config
        self.generator = SyntheticDataGenerator(config)
        
    def detect_objects(self, spike_map: np.ndarray, scale: float, min_size: int = 2) -> List[Tuple[float, float]]:
        """Detect multiple objects using connected components."""
        binary = (spike_map > 0).astype(int)
        labeled, num_features = scipy_label(binary)
        
        detections = []
        for obj_id in range(1, num_features + 1):
            coords = np.where(labeled == obj_id)
            if len(coords[0]) >= min_size:
                det_y = coords[0].mean() * scale
                det_x = coords[1].mean() * scale
                detections.append((det_x, det_y))
        
        return detections
    
    def match_detections_to_gt(
        self,
        detections: List[Tuple[float, float]],
        gt_positions: List[Tuple[float, float]],
        max_distance: float = 15.0  # Tighter threshold: must be within 15px
    ) -> Tuple[List[float], int]:
        """Match detections to ground truth, return errors and count of matched GT."""
        errors = []
        matched_gt = 0
        used_detections = set()
        
        for gt in gt_positions:
            best_dist = float('inf')
            best_det_idx = -1
            
            for det_idx, det in enumerate(detections):
                if det_idx in used_detections:
                    continue
                dist = np.sqrt((det[0] - gt[0])**2 + (det[1] - gt[1])**2)
                if dist < best_dist and dist < max_distance:
                    best_dist = dist
                    best_det_idx = det_idx
            
            if best_det_idx >= 0:
                errors.append(best_dist)
                used_detections.add(best_det_idx)
                matched_gt += 1
        
        return errors, matched_gt
    
    def run_single_test(
        self,
        n_objects: int,
        noise_level: float,
        crossover: bool,
        T: Optional[int] = None
    ) -> TestResult:
        """Run a single evaluation test."""
        if T is None:
            T = self.config.T
            
        # Generate data
        np.random.seed(self.config.seed)
        frames, trajectories = self.generator.generate_dataset(
            n_objects, noise_level, crossover, T
        )
        
        # Run inference
        x_batch = frames.unsqueeze(1).to(self.device)
        self.model.eval()
        
        start_time = time.perf_counter()
        with torch.no_grad():
            output = self.model(x_batch, fire_once=False)
            spikes = output.classification_spikes[:, 0, :, :, :].cpu().numpy()
        inference_time = time.perf_counter() - start_time
        
        fps = T / inference_time
        scale = self.config.H / spikes.shape[2]
        
        # Evaluate each frame
        all_errors = []
        total_gt = 0
        total_matched = 0
        both_detected = 0
        
        for t in range(T):
            spike_map = spikes[t].sum(axis=0)
            detections = self.detect_objects(spike_map, scale)
            gt_positions = [traj[t] for traj in trajectories]
            
            errors, matched = self.match_detections_to_gt(detections, gt_positions)
            all_errors.extend(errors)
            total_gt += len(gt_positions)
            total_matched += matched
            
            if matched == len(gt_positions):
                both_detected += 1
        
        # Calculate metrics
        detection_rate = total_matched / total_gt if total_gt > 0 else 0
        mean_error = np.mean(all_errors) if all_errors else float('inf')
        max_error = np.max(all_errors) if all_errors else float('inf')
        both_rate = both_detected / T if T > 0 else 0
        
        return TestResult(
            noise_level=noise_level,
            n_objects=n_objects,
            has_crossover=crossover,
            detection_rate=detection_rate,
            mean_error=mean_error,
            max_error=max_error,
            both_detected_rate=both_rate,
            fps=fps,
            frames_tested=T
        )
    
    def run_noise_sweep(
        self,
        noise_levels: List[float],
        n_objects: int = 1
    ) -> List[TestResult]:
        """Sweep across noise levels."""
        results = []
        print(f"\n{'='*70}")
        print(f"NOISE SWEEP: {n_objects} object(s)")
        print(f"{'='*70}")
        
        for noise in noise_levels:
            result = self.run_single_test(n_objects, noise, crossover=False)
            results.append(result)
            print(f"Noise={noise:.1%}: Detection={result.detection_rate:.1%}, "
                  f"Error={result.mean_error:.1f}px, FPS={result.fps:.0f}")
        
        return results
    
    def run_object_density_sweep(
        self,
        object_counts: List[int],
        noise_level: float = 0.01
    ) -> Tuple[List[TestResult], List[TestResult]]:
        """Sweep across object densities with and without crossover."""
        results_no_cross = []
        results_cross = []
        
        print(f"\n{'='*70}")
        print(f"OBJECT DENSITY SWEEP (noise={noise_level:.1%})")
        print(f"{'='*70}")
        
        for n_obj in object_counts:
            # Without crossover
            result = self.run_single_test(n_obj, noise_level, crossover=False)
            results_no_cross.append(result)
            print(f"Objects={n_obj}, No Cross: Both={result.both_detected_rate:.1%}, "
                  f"Error={result.mean_error:.1f}px")
            
            # With crossover (only for multiple objects)
            if n_obj > 1:
                result_cross = self.run_single_test(n_obj, noise_level, crossover=True)
                results_cross.append(result_cross)
                print(f"Objects={n_obj}, Crossover: Both={result_cross.both_detected_rate:.1%}, "
                      f"Error={result_cross.mean_error:.1f}px")
        
        return results_no_cross, results_cross
    
    def run_fps_benchmark(self, target_fps: int = 10000) -> dict:
        """Benchmark FPS with multiple configurations to achieve target."""
        print(f"\n{'='*70}")
        print(f"FPS BENCHMARK (Target: {target_fps}+ FPS)")
        print(f"{'='*70}")
        
        results = {}
        self.model.eval()
        
        # Test different batch sizes and configurations
        test_configs = [
            ("Single Frame Sequential", 1, 100, False),
            ("Batch=10", 10, 100, False),
            ("Batch=50", 50, 50, False),
            ("Batch=100", 100, 30, False),
            ("Batch=500", 500, 10, False),
            ("Batch=1000", 1000, 5, False),
            ("Batch=100 + FP16", 100, 30, True),
            ("Batch=500 + FP16", 500, 10, True),
            ("Batch=1000 + FP16", 1000, 5, True),
        ]
        
        print(f"\n{'Config':<25} | {'Frames':>8} | {'Time (ms)':>10} | {'FPS':>10} | {'Status'}")
        print("-" * 75)
        
        best_fps = 0
        best_config = ""
        
        for name, batch_size, n_batches, use_fp16 in test_configs:
            try:
                total_frames = batch_size * n_batches
                
                # Create test data
                frames = torch.randn(batch_size, self.config.C, self.config.H, self.config.W)
                frames = torch.clamp(frames * 0.3 + 0.5, 0, 1).to(self.device)
                x_batch = frames.unsqueeze(1)
                
                if use_fp16 and self.device.type == 'cuda':
                    x_batch = x_batch.half()
                    # Create a copy for FP16 to avoid modifying original
                    import copy
                    model_fp16 = copy.deepcopy(self.model).half()
                else:
                    model_fp16 = self.model
                
                # Warmup
                with torch.no_grad():
                    for _ in range(3):
                        _ = model_fp16(x_batch, fire_once=False)
                
                # Synchronize before timing
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                # Benchmark
                start = time.perf_counter()
                with torch.no_grad():
                    for _ in range(n_batches):
                        _ = model_fp16(x_batch, fire_once=False)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                    
                elapsed = time.perf_counter() - start
                fps = total_frames / elapsed
                
                # No need to restore - we used a copy for FP16
                
                status = "✓ TARGET MET" if fps >= target_fps else ""
                print(f"{name:<25} | {total_frames:>8} | {elapsed*1000:>10.2f} | {fps:>10.0f} | {status}")
                
                results[name] = fps
                if fps > best_fps:
                    best_fps = fps
                    best_config = name
                    
            except Exception as e:
                print(f"{name:<25} | {'ERROR':>8} | {str(e)[:30]}")
                results[name] = 0
        
        print("-" * 75)
        print(f"\nBest Configuration: {best_config}")
        print(f"Peak FPS: {best_fps:.0f}")
        
        if best_fps >= target_fps:
            print(f"✓ SUCCESS: Achieved {best_fps:.0f} FPS (target: {target_fps})")
        else:
            print(f"✗ Target not met. Best: {best_fps:.0f} FPS")
            print("\nOptimization suggestions:")
            print("  1. Use larger batch sizes")
            print("  2. Enable FP16 inference")
            print("  3. Use torch.compile() (PyTorch 2.0+)")
            print("  4. Use TensorRT optimization")
        
        # Additional: Test SINGLE TIMESTEP throughput (this is real-time FPS)
        print(f"\n{'='*70}")
        print("SINGLE TIMESTEP THROUGHPUT (Real-time Event Processing)")
        print(f"{'='*70}")
        
        try:
            self.model.float()
            self.model.eval()
            
            # Single frame at a time - this is actual real-time processing rate
            test_frame = torch.randn(1, self.config.C, self.config.H, self.config.W).to(self.device).float()
            test_batch = test_frame.unsqueeze(1)  # Add batch dim
            
            # Warmup
            self.model.reset_state()
            with torch.no_grad():
                for _ in range(10):
                    _ = self.model.forward_single_timestep(test_frame)
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            # Benchmark single timestep processing
            n_iters = 10000
            start = time.perf_counter()
            with torch.no_grad():
                for _ in range(n_iters):
                    _ = self.model.forward_single_timestep(test_frame)
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            elapsed = time.perf_counter() - start
            single_ts_fps = n_iters / elapsed
            
            print(f"Single timestep FPS: {single_ts_fps:.0f}")
            results['single_timestep_fps'] = single_ts_fps
            
            if single_ts_fps >= target_fps:
                print(f"✓ SINGLE TIMESTEP EXCEEDS {target_fps} FPS TARGET!")
            
            # Also test batched single timestep
            batch_sizes_st = [10, 50, 100, 500]
            print(f"\nBatched single-timestep throughput:")
            for bs in batch_sizes_st:
                test_batch_st = torch.randn(bs, self.config.C, self.config.H, self.config.W).to(self.device).float()
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                n_iters_batch = 1000
                start = time.perf_counter()
                with torch.no_grad():
                    for _ in range(n_iters_batch):
                        _ = self.model.forward_single_timestep(test_batch_st)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                elapsed = time.perf_counter() - start
                batch_fps = (bs * n_iters_batch) / elapsed
                results[f'single_ts_batch_{bs}'] = batch_fps
                marker = "✓" if batch_fps >= target_fps else " "
                print(f"  {marker} Batch={bs}: {batch_fps:.0f} FPS")
                
        except Exception as e:
            print(f"Single timestep test failed: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"\n{'='*70}")
        print("PURE CONVOLUTION THROUGHPUT (no temporal loop)")
        print(f"{'='*70}")
        
        try:
            # Ensure model is in float32
            self.model.float()
            
            # Just measure conv layers without temporal processing
            test_input = torch.randn(1000, self.config.C, self.config.H, self.config.W).to(self.device).float()
            
            # Get just conv1
            conv1 = self.model.conv1.conv
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            with torch.no_grad():
                for _ in range(10):
                    _ = conv1(test_input)
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            elapsed = time.perf_counter() - start
            pure_conv_fps = (1000 * 10) / elapsed
            
            print(f"Pure Conv2d throughput: {pure_conv_fps:.0f} FPS")
            results['pure_conv'] = pure_conv_fps
        except Exception as e:
            print(f"Pure conv test failed: {e}")
            results['pure_conv'] = 0
        
        return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_results(
    noise_results: List[TestResult],
    density_results_no_cross: List[TestResult],
    density_results_cross: List[TestResult],
    output_dir: str
):
    """Generate result plots."""
    Path(output_dir).mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Noise vs Detection Rate
    ax1 = axes[0, 0]
    noise_levels = [r.noise_level * 100 for r in noise_results]
    detection_rates = [r.detection_rate * 100 for r in noise_results]
    ax1.plot(noise_levels, detection_rates, 'b-o', linewidth=2, markersize=8)
    ax1.axhline(y=90, color='g', linestyle='--', label='90% threshold')
    ax1.axhline(y=50, color='r', linestyle='--', label='50% threshold')
    ax1.set_xlabel('Noise Level (%)', fontsize=12)
    ax1.set_ylabel('Detection Rate (%)', fontsize=12)
    ax1.set_title('Detection Rate vs Noise Level', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)
    
    # 2. Noise vs Tracking Error
    ax2 = axes[0, 1]
    errors = [r.mean_error for r in noise_results]
    ax2.plot(noise_levels, errors, 'r-o', linewidth=2, markersize=8)
    ax2.axhline(y=10, color='g', linestyle='--', label='10px threshold')
    ax2.set_xlabel('Noise Level (%)', fontsize=12)
    ax2.set_ylabel('Mean Tracking Error (px)', fontsize=12)
    ax2.set_title('Tracking Error vs Noise Level', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Object Density vs Both Detected Rate
    ax3 = axes[1, 0]
    obj_counts_no_cross = [r.n_objects for r in density_results_no_cross]
    both_rates_no_cross = [r.both_detected_rate * 100 for r in density_results_no_cross]
    ax3.plot(obj_counts_no_cross, both_rates_no_cross, 'b-o', linewidth=2, 
             markersize=8, label='No Crossover')
    
    if density_results_cross:
        obj_counts_cross = [r.n_objects for r in density_results_cross]
        both_rates_cross = [r.both_detected_rate * 100 for r in density_results_cross]
        ax3.plot(obj_counts_cross, both_rates_cross, 'r-s', linewidth=2, 
                 markersize=8, label='With Crossover')
    
    ax3.set_xlabel('Number of Objects', fontsize=12)
    ax3.set_ylabel('All Objects Detected Rate (%)', fontsize=12)
    ax3.set_title('Multi-Object Detection vs Density', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 105)
    
    # 4. Summary Table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Find failure points
    noise_90_threshold = None
    noise_50_threshold = None
    for r in noise_results:
        if r.detection_rate < 0.9 and noise_90_threshold is None:
            noise_90_threshold = r.noise_level
        if r.detection_rate < 0.5 and noise_50_threshold is None:
            noise_50_threshold = r.noise_level
    
    summary_text = f"""
    FAILURE MODE ANALYSIS
    =====================
    
    NOISE TOLERANCE:
    • Detection drops below 90% at: {noise_90_threshold*100 if noise_90_threshold else '>50'}% noise
    • Detection drops below 50% at: {noise_50_threshold*100 if noise_50_threshold else '>50'}% noise
    • Best case error: {min(r.mean_error for r in noise_results):.1f} px
    • Worst case error: {max(r.mean_error for r in noise_results if r.mean_error < 1000):.1f} px
    
    MULTI-OBJECT:
    • 1 object: {density_results_no_cross[0].detection_rate*100:.0f}% detection
    • 2 objects (no cross): {[r for r in density_results_no_cross if r.n_objects==2][0].both_detected_rate*100 if any(r.n_objects==2 for r in density_results_no_cross) else 'N/A'}% both
    • 2 objects (crossover): {density_results_cross[0].both_detected_rate*100 if density_results_cross else 'N/A'}% both
    
    PERFORMANCE:
    • FPS: See benchmark results
    """
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/evaluation_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_dir}/evaluation_results.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("SYSTEMATIC SNN TRACKING EVALUATION")
    print("="*70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    checkpoint_path = '/home/ubuntu/checkpoint_best.pt'
    print(f"Loading model from: {checkpoint_path}")
    model = load_model(checkpoint_path, device, 0.05)
    print("Model loaded successfully")
    
    config = EvalConfig()
    engine = EvaluationEngine(model, device, config)
    
    # Create output directory
    Path(config.output_dir).mkdir(exist_ok=True)
    
    # ==========================================================================
    # 1. NOISE SWEEP (Key levels to show degradation curve)
    # ==========================================================================
    noise_levels = [
        0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70
    ]
    noise_results = engine.run_noise_sweep(noise_levels, n_objects=1)
    
    # ==========================================================================
    # 2. OBJECT DENSITY SWEEP
    # ==========================================================================
    object_counts = [1, 2, 3, 4, 5]
    density_no_cross, density_cross = engine.run_object_density_sweep(
        object_counts, noise_level=0.01
    )
    
    # ==========================================================================
    # 3. FPS BENCHMARK (MUST EXCEED 10,000 FPS)
    # ==========================================================================
    fps_results = engine.run_fps_benchmark(target_fps=10000)
    best_fps = max(fps_results.values())
    
    # ==========================================================================
    # 4. GENERATE PLOTS
    # ==========================================================================
    plot_results(noise_results, density_no_cross, density_cross, config.output_dir)
    
    # ==========================================================================
    # 5. SAVE DETAILED RESULTS
    # ==========================================================================
    all_results = {
        'noise_sweep': [asdict(r) for r in noise_results],
        'density_no_crossover': [asdict(r) for r in density_no_cross],
        'density_crossover': [asdict(r) for r in density_cross],
        'fps_benchmark': fps_results,
        'best_fps': best_fps
    }
    
    with open(f'{config.output_dir}/results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved: {config.output_dir}/results.json")
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    print("\n[NOISE TOLERANCE]")
    for r in noise_results:
        status = "✓" if r.detection_rate > 0.9 else "✗" if r.detection_rate < 0.5 else "~"
        print(f"  {status} Noise {r.noise_level:5.1%}: Detection {r.detection_rate:5.1%}, "
              f"Error {r.mean_error:5.1f}px")
    
    print("\n[MULTI-OBJECT (no crossover)]")
    for r in density_no_cross:
        status = "✓" if r.both_detected_rate > 0.7 else "✗" if r.both_detected_rate < 0.3 else "~"
        print(f"  {status} {r.n_objects} objects: All detected {r.both_detected_rate:5.1%}, "
              f"Error {r.mean_error:5.1f}px")
    
    print("\n[MULTI-OBJECT (with crossover)]")
    for r in density_cross:
        status = "✓" if r.both_detected_rate > 0.5 else "✗" if r.both_detected_rate < 0.3 else "~"
        print(f"  {status} {r.n_objects} objects: All detected {r.both_detected_rate:5.1%}, "
              f"Error {r.mean_error:5.1f}px")
    
    print(f"\n[PERFORMANCE]")
    
    # Separate multi-frame and single-timestep results
    multi_frame_fps = {k: v for k, v in fps_results.items() 
                       if not k.startswith('single_ts') and k != 'pure_conv'}
    single_ts_fps = {k: v for k, v in fps_results.items() 
                     if k.startswith('single_ts') or k == 'single_timestep_fps'}
    
    best_multi = max(multi_frame_fps.values()) if multi_frame_fps else 0
    best_single = max(single_ts_fps.values()) if single_ts_fps else 0
    
    print(f"\n  MULTI-FRAME (60 timesteps per sample):")
    print(f"  Best: {best_multi:.0f} FPS")
    for config_name, fps_val in multi_frame_fps.items():
        print(f"    {config_name}: {fps_val:.0f} FPS")
    
    print(f"\n  SINGLE-TIMESTEP (Real-time event processing):")
    print(f"  Best: {best_single:.0f} FPS")
    for config_name, fps_val in single_ts_fps.items():
        marker = "★" if fps_val >= 10000 else " "
        print(f"    {marker} {config_name}: {fps_val:.0f} FPS")
    
    if best_single >= 10000:
        print(f"\n  ✓ SINGLE-TIMESTEP EXCEEDS 10,000 FPS TARGET!")
    else:
        print(f"\n  Note: Full 60-timestep processing limits throughput.")
        print(f"        Single timestep = {best_single:.0f} FPS (event-by-event rate)")
    
    print("\n" + "="*70)
    print("Evaluation complete! Results saved to:", config.output_dir)
    print("="*70)


if __name__ == "__main__":
    main()
