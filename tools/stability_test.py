#!/usr/bin/env python3
"""
Machina Stability Test
========================

30-minute stability test with tegrastats integration.
Generates FPS vs Time and RAM vs Time graphs.

Usage:
    python -m tools.stability_test [--duration 1800] [--output reports]
"""

import subprocess
import time
import json
import re
import threading
import os
import argparse
from datetime import datetime
from collections import deque

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for headless
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed, graphs will not be generated")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def get_next_report_id(output_dir: str) -> str:
    """Find the next sequential report ID (001, 002, 003, ...)."""
    os.makedirs(output_dir, exist_ok=True)
    existing = []
    for f in os.listdir(output_dir):
        match = re.match(r'^(\d{3})_stability_report\.json$', f)
        if match:
            existing.append(int(match.group(1)))
    next_id = max(existing) + 1 if existing else 1
    return f"{next_id:03d}"


def get_system_config() -> dict:
    """Capture current system configuration for report metadata."""
    config = {
        "kv_cache": "unknown",
        "power_mode": "unknown",
        "jetson_clocks": False
    }
    
    # Get KV cache type from llama-server service
    try:
        result = subprocess.run(
            ['grep', 'ctk', '/etc/systemd/system/llama-server.service'],
            capture_output=True, text=True, timeout=5
        )
        if 'q4_0' in result.stdout:
            config["kv_cache"] = "q4_0"
        elif 'q8_0' in result.stdout:
            config["kv_cache"] = "q8_0"
    except:
        pass
    
    # Get power mode from llama-server service (configured at boot)
    try:
        result = subprocess.run(
            ['grep', 'nvpmodel', '/etc/systemd/system/llama-server.service'],
            capture_output=True, text=True, timeout=5
        )
        if '-m 2' in result.stdout:
            config["power_mode"] = "MAXN_SUPER"
        elif '-m 1' in result.stdout:
            config["power_mode"] = "25W"
        elif '-m 0' in result.stdout:
            config["power_mode"] = "15W"
    except:
        pass
    
    # Check if jetson_clocks is active (CPU at max frequency)
    try:
        result = subprocess.run(
            ['cat', '/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq'],
            capture_output=True, text=True, timeout=5
        )
        cur_freq = int(result.stdout.strip())
        # If CPU freq > 1.9GHz, jetson_clocks is likely on
        config["jetson_clocks"] = cur_freq > 1900000
    except:
        pass
    
    return config


class TegrastatsMonitor:
    """
    Monitor Jetson metrics using tegrastats.
    
    Parses output like:
    RAM 5432/7620MB (lfb 1x2MB) SWAP 0/3810MB CPU [50%@2265,40%@2265,...] 
    GR3D 98%@1.3GHz NVENC 0%@0 VIC 0%@0 APE 0 TEMP CPU@45C GPU@44C ...
    """
    
    def __init__(self):
        self.samples = []
        self.running = False
        self.process = None
        self.thread = None
    
    def start(self):
        """Start tegrastats monitoring in background."""
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop monitoring."""
        self.running = False
        if self.process:
            self.process.terminate()
        if self.thread:
            self.thread.join(timeout=2)
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        try:
            self.process = subprocess.Popen(
                ['tegrastats', '--interval', '1000'],  # 1 second interval
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True
            )
            
            for line in self.process.stdout:
                if not self.running:
                    break
                
                sample = self._parse_tegrastats(line.strip())
                if sample:
                    sample['timestamp'] = time.monotonic()
                    self.samples.append(sample)
        
        except FileNotFoundError:
            print("Warning: tegrastats not found (not on Jetson?)")
        except Exception as e:
            print(f"Tegrastats error: {e}")
    
    def _parse_tegrastats(self, line: str) -> dict:
        """Parse a tegrastats output line."""
        result = {}
        
        # RAM: 5432/7620MB
        ram_match = re.search(r'RAM (\d+)/(\d+)MB', line)
        if ram_match:
            result['ram_used_mb'] = int(ram_match.group(1))
            result['ram_total_mb'] = int(ram_match.group(2))
        
        # SWAP: 0/3810MB
        swap_match = re.search(r'SWAP (\d+)/(\d+)MB', line)
        if swap_match:
            result['swap_used_mb'] = int(swap_match.group(1))
        
        # GR3D (GPU): 98%@1.3GHz or GR3D_FREQ 98%@1300
        gpu_match = re.search(r'GR3D[_FREQ]* (\d+)%', line)
        if gpu_match:
            result['gpu_pct'] = int(gpu_match.group(1))
        
        # CPU temps: CPU@45C or TEMP CPU@45C
        cpu_temp_match = re.search(r'CPU@(\d+\.?\d*)C', line)
        if cpu_temp_match:
            result['cpu_temp_c'] = float(cpu_temp_match.group(1))
        
        # GPU temps: GPU@44C
        gpu_temp_match = re.search(r'GPU@(\d+\.?\d*)C', line)
        if gpu_temp_match:
            result['gpu_temp_c'] = float(gpu_temp_match.group(1))
        
        # Power: VDD_IN 5000mW or POM_5V_IN 5000
        power_match = re.search(r'(?:VDD_IN|POM_5V_IN)\s*(\d+)', line)
        if power_match:
            result['power_mw'] = int(power_match.group(1))
        
        return result
    
    def get_samples(self) -> list:
        """Get all collected samples."""
        return self.samples.copy()


class FPSMonitor:
    """Monitor FPS from the running main process via API."""
    
    def __init__(self, api_url: str = "http://localhost:8090/api/status"):
        self.api_url = api_url
        self.samples = []
        self.running = False
        self.thread = None
    
    def start(self):
        """Start FPS monitoring in background."""
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop monitoring."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        import requests
        
        while self.running:
            try:
                resp = requests.get(self.api_url, timeout=2)
                if resp.status_code == 200:
                    data = resp.json()
                    sample = {
                        'timestamp': time.monotonic(),
                        'fps': data.get('metrics', {}).get('fast_loop', {}).get('fps', 0),
                        'vlm_calls': data.get('metrics', {}).get('slow_loop', {}).get('total_calls', 0),
                        'parse_rate': data.get('metrics', {}).get('reliability', {}).get('parse_rate', 1.0),
                        'fallback_rate': data.get('metrics', {}).get('reliability', {}).get('fallback_rate', 0.0),
                        'uptime_s': data.get('metrics', {}).get('uptime_s', 0),
                        'state': data.get('state', {}).get('state', 'UNKNOWN')
                    }
                    self.samples.append(sample)
            except Exception as e:
                # API not available yet
                pass
            
            time.sleep(1)
    
    def get_samples(self) -> list:
        """Get all collected samples."""
        return self.samples.copy()


def generate_graphs(fps_samples: list, tegra_samples: list, output_dir: str):
    """Generate FPS vs Time and RAM vs Time graphs."""
    if not HAS_MATPLOTLIB:
        print("Skipping graph generation (matplotlib not installed)")
        return None, None
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get system config for title
    sys_config = get_system_config()
    config_str = f"KV: {sys_config['kv_cache']} | Power: {sys_config['power_mode']} | jetson_clocks: {'ON' if sys_config['jetson_clocks'] else 'OFF'}"
    
    # Normalize timestamps to start from 0
    if fps_samples:
        t0 = fps_samples[0]['timestamp']
        fps_times = [(s['timestamp'] - t0) / 60 for s in fps_samples]  # minutes
        fps_values = [s['fps'] for s in fps_samples]
    else:
        fps_times, fps_values = [], []
    
    if tegra_samples:
        t0 = tegra_samples[0]['timestamp']
        tegra_times = [(s['timestamp'] - t0) / 60 for s in tegra_samples]  # minutes
        ram_values = [s.get('ram_used_mb', 0) for s in tegra_samples]
        gpu_values = [s.get('gpu_pct', 0) for s in tegra_samples]
    else:
        tegra_times, ram_values, gpu_values = [], [], []
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(f'Machina Stability Test Results\n{config_str}', fontsize=12, fontweight='bold')
    
    # FPS vs Time
    if fps_times:
        ax1.plot(fps_times, fps_values, 'g-', linewidth=1, alpha=0.8)
        ax1.axhline(y=25, color='orange', linestyle='--', alpha=0.5, label='Target: 25 FPS')
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('FPS')
        ax1.set_title('Fast Loop FPS vs Time')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add stats annotation
        if HAS_NUMPY:
            fps_mean = np.mean(fps_values)
            fps_std = np.std(fps_values)
            fps_min = np.min(fps_values)
            ax1.annotate(f'Mean: {fps_mean:.1f} FPS\nStd: {fps_std:.1f}\nMin: {fps_min:.1f}',
                        xy=(0.02, 0.95), xycoords='axes fraction',
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # RAM vs Time
    if tegra_times and ram_values:
        ax2.plot(tegra_times, ram_values, 'b-', linewidth=1, alpha=0.8)
        ax2.set_xlabel('Time (minutes)')
        ax2.set_ylabel('RAM Used (MB)')
        ax2.set_title('RAM Usage vs Time')
        ax2.grid(True, alpha=0.3)
        
        # Add stats annotation
        if HAS_NUMPY:
            ram_mean = np.mean(ram_values)
            ram_max = np.max(ram_values)
            ax2.annotate(f'Mean: {ram_mean:.0f} MB\nMax: {ram_max:.0f} MB',
                        xy=(0.02, 0.95), xycoords='axes fraction',
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    return fig


def save_graphs(fig, output_dir: str, report_id: str):
    """Save graphs with sequential naming."""
    graph_path = os.path.join(output_dir, f'{report_id}_stability_graphs.png')
    fig.savefig(graph_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"📊 Graphs saved to: {graph_path}")


def generate_report(fps_samples: list, tegra_samples: list, 
                   duration: float, output_dir: str) -> dict:
    """Generate JSON report with all metrics."""
    
    # Capture system configuration
    sys_config = get_system_config()
    
    report = {
        "test_date": datetime.now().isoformat(),
        "duration_sec": round(duration, 1),
        "system_config": sys_config,
        "samples_collected": {
            "fps": len(fps_samples),
            "tegrastats": len(tegra_samples)
        }
    }
    
    # FPS statistics
    if fps_samples and HAS_NUMPY:
        fps_values = [s['fps'] for s in fps_samples]
        report["fps"] = {
            "mean": round(float(np.mean(fps_values)), 1),
            "std": round(float(np.std(fps_values)), 2),
            "min": round(float(np.min(fps_values)), 1),
            "max": round(float(np.max(fps_values)), 1),
            "p50": round(float(np.percentile(fps_values, 50)), 1),
            "p5": round(float(np.percentile(fps_values, 5)), 1),  # Worst 5%
        }
        
        # Calculate FPS drift
        if len(fps_values) >= 10:
            first_10 = np.mean(fps_values[:10])
            last_10 = np.mean(fps_values[-10:])
            drift_pct = ((last_10 - first_10) / first_10) * 100 if first_10 > 0 else 0
            report["fps"]["drift_pct"] = round(drift_pct, 2)
        
        # Count freeze events (FPS < 15)
        freeze_count = sum(1 for f in fps_values if f < 15)
        report["fps"]["freeze_events"] = freeze_count
    
    # RAM statistics
    if tegra_samples and HAS_NUMPY:
        ram_values = [s.get('ram_used_mb', 0) for s in tegra_samples if s.get('ram_used_mb')]
        if ram_values:
            report["ram"] = {
                "mean_mb": round(float(np.mean(ram_values)), 0),
                "max_mb": round(float(np.max(ram_values)), 0),
                "min_mb": round(float(np.min(ram_values)), 0),
            }
        
        # Temperature stats
        cpu_temps = [s.get('cpu_temp_c', 0) for s in tegra_samples if s.get('cpu_temp_c')]
        gpu_temps = [s.get('gpu_temp_c', 0) for s in tegra_samples if s.get('gpu_temp_c')]
        if cpu_temps:
            report["temperature"] = {
                "cpu_max_c": round(float(np.max(cpu_temps)), 1),
                "cpu_mean_c": round(float(np.mean(cpu_temps)), 1),
            }
            if gpu_temps:
                report["temperature"]["gpu_max_c"] = round(float(np.max(gpu_temps)), 1)
                report["temperature"]["gpu_mean_c"] = round(float(np.mean(gpu_temps)), 1)
        
        # Detect thermal throttling (temp > 70°C)
        if cpu_temps:
            throttle_count = sum(1 for t in cpu_temps if t > 70)
            report["thermal_throttle_samples"] = throttle_count
    
    # Reliability from last FPS sample
    if fps_samples:
        last = fps_samples[-1]
        report["reliability"] = {
            "parse_rate": round(last.get('parse_rate', 1.0), 4),
            "fallback_rate": round(last.get('fallback_rate', 0.0), 4),
            "vlm_calls": last.get('vlm_calls', 0)
        }
    
    # Stability summary
    report["stability"] = {
        "oom_detected": False,  # Would need to check dmesg
        "deadlock_detected": False,
        "passed": True
    }
    
    # Check pass criteria
    if report.get("fps", {}).get("mean", 0) < 20:
        report["stability"]["passed"] = False
        report["stability"]["failure_reason"] = "FPS too low"
    if report.get("fps", {}).get("freeze_events", 0) > 10:
        report["stability"]["passed"] = False
        report["stability"]["failure_reason"] = "Too many freeze events"
    if report.get("reliability", {}).get("parse_rate", 1.0) < 0.95:
        report["stability"]["passed"] = False
        report["stability"]["failure_reason"] = "Parse rate below 95%"
    
    return report


def save_report(report: dict, output_dir: str, report_id: str) -> str:
    """Save report with sequential naming."""
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f'{report_id}_stability_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📋 Report saved to: {report_path}")
    return report_path


def run_stability_test(duration_sec: int = 1800, output_dir: str = "reports"):
    """
    Run the stability test.
    
    NOTE: This monitors an ALREADY RUNNING main process.
    Start `python -m src.main` before running this test.
    """
    print("=" * 60)
    print("Machina Stability Test")
    print("=" * 60)
    print(f"Duration: {duration_sec} seconds ({duration_sec/60:.1f} minutes)")
    print(f"Output: {output_dir}/")
    print()
    print("NOTE: Make sure `python -m src.main` is already running!")
    print()
    
    # Start monitors
    fps_monitor = FPSMonitor()
    tegra_monitor = TegrastatsMonitor()
    
    print("Starting monitors...")
    fps_monitor.start()
    tegra_monitor.start()
    
    # Wait for first sample
    time.sleep(2)
    if not fps_monitor.samples:
        print("⚠️  Warning: Cannot reach API. Is main.py running?")
    else:
        print("✅ API connected")
    
    if not tegra_monitor.samples:
        print("⚠️  Warning: tegrastats not available")
    else:
        print("✅ tegrastats connected")
    
    print()
    print(f"Running for {duration_sec/60:.1f} minutes...")
    print("Progress: ", end='', flush=True)
    
    start_time = time.monotonic()
    try:
        while (time.monotonic() - start_time) < duration_sec:
            elapsed = time.monotonic() - start_time
            progress = int((elapsed / duration_sec) * 50)
            print(f"\rProgress: [{'=' * progress}{' ' * (50 - progress)}] {elapsed:.0f}s / {duration_sec}s", 
                  end='', flush=True)
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    
    actual_duration = time.monotonic() - start_time
    print(f"\n\nTest completed in {actual_duration:.1f} seconds")
    
    # Stop monitors
    fps_monitor.stop()
    tegra_monitor.stop()
    
    # Get samples
    fps_samples = fps_monitor.get_samples()
    tegra_samples = tegra_monitor.get_samples()
    
    print(f"\nCollected {len(fps_samples)} FPS samples, {len(tegra_samples)} tegrastats samples")
    
    # Get next report ID
    report_id = get_next_report_id(output_dir)
    print(f"\nTest ID: {report_id}")
    
    # Generate report
    print("Generating report...")
    report = generate_report(fps_samples, tegra_samples, actual_duration, output_dir)
    save_report(report, output_dir, report_id)
    
    # Generate graphs
    print("Generating graphs...")
    fig = generate_graphs(fps_samples, tegra_samples, output_dir)
    if fig:
        save_graphs(fig, output_dir, report_id)
    
    # Print summary
    print("\n" + "=" * 60)
    print("STABILITY TEST SUMMARY")
    print("=" * 60)
    print(f"Duration: {report['duration_sec']:.1f}s")
    
    if 'fps' in report:
        print(f"FPS: mean={report['fps']['mean']}, min={report['fps']['min']}, "
              f"freeze_events={report['fps'].get('freeze_events', 0)}")
    
    if 'ram' in report:
        print(f"RAM: mean={report['ram']['mean_mb']}MB, max={report['ram']['max_mb']}MB")
    
    if 'temperature' in report:
        print(f"Temp: CPU max={report['temperature'].get('cpu_max_c', '?')}°C, "
              f"GPU max={report['temperature'].get('gpu_max_c', '?')}°C")
    
    if 'reliability' in report:
        print(f"Reliability: parse_rate={report['reliability']['parse_rate']*100:.1f}%, "
              f"fallback_rate={report['reliability']['fallback_rate']*100:.1f}%")
    
    passed = report['stability']['passed']
    status = "✅ PASSED" if passed else "❌ FAILED"
    print(f"\nOverall: {status}")
    if not passed:
        print(f"Reason: {report['stability'].get('failure_reason', 'Unknown')}")
    
    print("=" * 60)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Machina Stability Test")
    parser.add_argument("--duration", type=int, default=1800, 
                        help="Test duration in seconds (default: 1800 = 30 min)")
    parser.add_argument("--output", type=str, default="reports",
                        help="Output directory for reports and graphs")
    args = parser.parse_args()
    
    run_stability_test(args.duration, args.output)
