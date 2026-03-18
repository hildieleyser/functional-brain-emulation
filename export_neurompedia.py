"""
Export multi-session neurobehavioral data for the Neurompedia 3D demo.
Processes multiple XDF recordings across subjects, extracts signals,
computes spectral features, correlations, and session metadata.
Also extracts video thumbnails where available.
"""

import sys
sys.executable  # Using the right Python
import numpy as np
import pyxdf
import json
import base64
import os
import glob
from scipy.signal import butter, sosfilt, welch
from scipy.stats import pearsonr
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("WARNING: cv2 not available, skipping video frame extraction")

DATA_DIR = r"C:\Users\chris\Documents\CurrentStudy"
OUT_DIR  = r"C:\Users\chris\LSL_Tools\neurompedia"
os.makedirs(OUT_DIR, exist_ok=True)

# All known stream → modality mappings
MODALITY_DEFS = {
    "Gaze":        {"streams": ["Gazepoint", "TobiiGlasses_Gaze", "PupilNeon_Gaze"],
                    "max_ch": 6, "color": "#00E5FF", "desc": "Eye Position & Fixation"},
    "fNIRS-Oxy":   {"streams": ["KernelFlow_Red"],
                    "max_ch": 12, "color": "#FF1744", "desc": "Hemoglobin Oxygenation"},
    "fNIRS-Deoxy": {"streams": ["KernelFlow_IR"],
                    "max_ch": 12, "color": "#E040FB", "desc": "Hemoglobin Deoxygenation"},
    "fNIRS-QA":    {"streams": ["KernelFlow_Quality"],
                    "max_ch": 8, "color": "#FF9100", "desc": "Signal Quality Index"},
    "EEG":         {"streams": ["DSI24"],
                    "max_ch": 24, "color": "#7C4DFF", "desc": "Electroencephalography"},
    "Hand-L":      {"streams": ["StretchSense"],
                    "max_ch": 7, "color": "#76FF03", "desc": "Left Hand Kinematics"},
    "Hand-R":      {"streams": ["StretchSense"],
                    "max_ch": 7, "color": "#00E676", "desc": "Right Hand Kinematics"},
    "Body":        {"streams": ["StretchSense"],
                    "max_ch": 7, "color": "#FFD600", "desc": "Body Motion Capture"},
    "IMU":         {"streams": ["TobiiGlasses_IMU", "PupilNeon_IMU"],
                    "max_ch": 9, "color": "#FF6D00", "desc": "Inertial Measurement"},
    "Audio":       {"streams": ["Microphone"],
                    "max_ch": 1, "color": "#448AFF", "desc": "Voice & Audio"},
    "EMG":         {"streams": ["Delsys_EMG"],
                    "max_ch": 16, "color": "#F50057", "desc": "Electromyography"},
}

# Channel index mappings for shared streams (StretchSense)
STRETCH_SENSE_MAP = {
    "Hand-L": list(range(0, 56, 8)),
    "Hand-R": list(range(56, 112, 8)),
    "Body":   list(range(112, 224, 16)),
}

SR = 10  # Resample to 10 Hz for richer signal detail
EXPORT_DURATION = 120  # seconds per session (max)

def find_xdf_files():
    """Find all XDF files and group by subject/session."""
    files = []
    for pattern in ["session_*.xdf", "sub-*_*.xdf"]:
        files.extend(glob.glob(os.path.join(DATA_DIR, pattern)))
    # Remove _old files for cleaner data
    files = [f for f in files if "_old" not in os.path.basename(f)]
    # Sort by size (largest first = most data)
    files.sort(key=lambda f: os.path.getsize(f), reverse=True)
    return files

def parse_session_info(filepath):
    """Extract subject/session metadata from filename."""
    basename = os.path.basename(filepath)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    info = {"file": filepath, "basename": basename, "size_mb": round(size_mb, 1)}
    if basename.startswith("sub-"):
        parts = basename.replace(".xdf", "").split("_")
        for p in parts:
            if p.startswith("sub-"): info["subject"] = p[4:]
            if p.startswith("ses-"): info["session"] = p[4:]
            if p.startswith("task-"): info["task"] = p[5:]
    elif basename.startswith("session_"):
        ts = basename.replace("session_", "").replace(".xdf", "")
        info["subject"] = "session"
        info["session"] = ts
        info["task"] = "Recording"
    return info

def resample_stream(stream, channels, target_t):
    """Resample a stream's channels to target timestamps."""
    ts = np.array(stream["time_stamps"])
    data = np.array(stream["time_series"], dtype=np.float64)
    mask = (ts >= target_t[0] - 1) & (ts <= target_t[-1] + 1)
    ts, data = ts[mask], data[mask]
    if len(ts) < 10:
        return np.zeros((len(target_t), len(channels)))
    result = np.zeros((len(target_t), len(channels)))
    for i, ch in enumerate(channels):
        ci = min(ch, data.shape[1] - 1)
        v = data[:, ci].astype(float)
        good = np.isfinite(v)
        if good.sum() < 10:
            continue
        result[:, i] = np.interp(target_t, ts[good], v[good])
    return result

def compute_band_power(data, sr, bands=None):
    """Compute spectral band power for signal data."""
    if bands is None:
        bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13),
                 "beta": (13, 30), "gamma": (30, 50)}
    result = {}
    for bname, (flo, fhi) in bands.items():
        flo_eff = min(flo, sr / 2 - 0.1)
        fhi_eff = min(fhi, sr / 2 - 0.1)
        if flo_eff >= fhi_eff:
            result[bname] = 0
            continue
        try:
            f, pxx = welch(data, fs=sr, nperseg=min(len(data), int(sr * 2)))
            idx = (f >= flo_eff) & (f <= fhi_eff)
            result[bname] = float(np.mean(pxx[idx])) if idx.any() else 0
        except Exception:
            result[bname] = 0
    return result

def process_xdf(filepath, max_duration=EXPORT_DURATION):
    """Process a single XDF file and extract all modality data."""
    print(f"  Loading {os.path.basename(filepath)}...")
    try:
        streams, header = pyxdf.load_xdf(filepath)
    except Exception as e:
        print(f"  ERROR loading: {e}")
        return None

    stream_map = {}
    for s in streams:
        name = s["info"]["name"][0]
        if len(s["time_stamps"]) > 10:
            stream_map[name] = s

    if not stream_map:
        print("  No usable streams found")
        return None

    # Find common time window
    all_starts = [s["time_stamps"][0] for s in stream_map.values()]
    all_ends = [s["time_stamps"][-1] for s in stream_map.values()]
    t_start = max(all_starts)
    t_end = min(min(all_ends), t_start + max_duration)
    duration = t_end - t_start

    if duration < 5:
        print(f"  Duration too short: {duration:.1f}s")
        return None

    n_samples = int(duration * SR)
    t_viz = np.linspace(t_start, t_end, n_samples)

    print(f"  Streams: {list(stream_map.keys())}")
    print(f"  Duration: {duration:.1f}s, Samples: {n_samples}")

    # Extract per-modality data
    modalities = {}
    for mod_name, mod_def in MODALITY_DEFS.items():
        # Find matching stream
        matched_stream = None
        for sname in mod_def["streams"]:
            if sname in stream_map:
                matched_stream = sname
                break
        if not matched_stream:
            continue

        # Determine channels
        if mod_name in STRETCH_SENSE_MAP:
            channels = STRETCH_SENSE_MAP[mod_name]
            max_avail = stream_map[matched_stream]["time_series"].shape[1] if len(stream_map[matched_stream]["time_series"]) > 0 else 0
            channels = [c for c in channels if c < max_avail]
        else:
            n_ch = min(mod_def["max_ch"],
                       stream_map[matched_stream]["time_series"].shape[1] if len(stream_map[matched_stream]["time_series"]) > 0 else 0)
            channels = list(range(n_ch))

        if not channels:
            continue

        # Resample
        d = resample_stream(stream_map[matched_stream], channels, t_viz)

        # Normalize per channel (z-score then sigmoid)
        for c in range(d.shape[1]):
            mu, sd = np.nanmean(d[:, c]), np.nanstd(d[:, c])
            if sd > 1e-10:
                d[:, c] = (d[:, c] - mu) / sd
        d_sig = 1.0 / (1.0 + np.exp(-np.clip(d, -6, 6)))

        # Mean signal
        mean_sig = np.nanmean(d_sig, axis=1)

        # Compute spectral features for EEG if available
        spectral = None
        if mod_name == "EEG" and d.shape[1] > 0:
            # Compute band power over sliding windows
            win_samples = min(SR * 4, n_samples)
            spectral_series = []
            for t_idx in range(0, n_samples, max(1, n_samples // 60)):
                t0 = max(0, t_idx - win_samples)
                seg = d[t0:t_idx + 1, 0]  # Use first channel
                if len(seg) > SR:
                    bp = compute_band_power(seg, SR)
                    spectral_series.append(bp)
                else:
                    spectral_series.append({"delta": 0, "theta": 0, "alpha": 0, "beta": 0, "gamma": 0})
            spectral = spectral_series

        modalities[mod_name] = {
            "color": mod_def["color"],
            "desc": mod_def["desc"],
            "nChannels": len(channels),
            "mean": [round(float(v), 4) for v in mean_sig],
            "channels": [[round(float(v), 3) for v in row] for row in d_sig.tolist()],
            "spectral": spectral,
        }

    if not modalities:
        print("  No modalities extracted")
        return None

    # Correlations (rolling window)
    mod_names = list(modalities.keys())
    n_mod = len(mod_names)
    WIN = max(10, SR * 4)
    corr_series = []
    for t in range(n_samples):
        t0 = max(0, t - WIN)
        mat = {}
        if t - t0 >= 5:
            for i in range(n_mod):
                for j in range(i + 1, n_mod):
                    a = np.array(modalities[mod_names[i]]["mean"][t0:t + 1])
                    b = np.array(modalities[mod_names[j]]["mean"][t0:t + 1])
                    if np.std(a) > 1e-10 and np.std(b) > 1e-10:
                        r, _ = pearsonr(a, b)
                        mat[f"{mod_names[i]}|{mod_names[j]}"] = round(abs(float(r)), 3)
        corr_series.append(mat)

    return {
        "sampleRate": SR,
        "duration": round(duration, 1),
        "nSamples": n_samples,
        "streams": list(stream_map.keys()),
        "modalities": modalities,
        "correlations": corr_series,
    }

def extract_video_frames(video_path, n_frames=40, size=(192, 144)):
    """Extract thumbnail frames from video file."""
    if not HAS_CV2 or not os.path.exists(video_path):
        return []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return []
    step = max(1, total // n_frames)
    frames = []
    for i in range(n_frames):
        fi = min(i * step, total - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            frames.append("")
            continue
        frame = cv2.resize(frame, size)
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
        frames.append(base64.b64encode(buf).decode('ascii'))
    cap.release()
    return frames

# -- Main ------------------------------------------------------------------
def main():
    print("=" * 60)
    print("NEUROMPEDIA DATA EXPORT")
    print("=" * 60)

    xdf_files = find_xdf_files()
    print(f"\nFound {len(xdf_files)} XDF files\n")

    # Process top sessions (by file size = most data)
    sessions = []
    total_size = 0
    total_channels = 0
    total_duration = 0
    all_modalities_seen = set()

    # Process up to 8 largest sessions
    for filepath in xdf_files[:8]:
        info = parse_session_info(filepath)
        print(f"\n{'-' * 50}")
        print(f"Processing: {info['basename']} ({info['size_mb']} MB)")

        result = process_xdf(filepath)
        if result is None:
            continue

        session_data = {
            "info": info,
            "data": result,
        }
        sessions.append(session_data)
        total_size += info["size_mb"]
        total_duration += result["duration"]
        for m in result["modalities"]:
            all_modalities_seen.add(m)
            total_channels += result["modalities"][m]["nChannels"]

        print(f"  -> {len(result['modalities'])} modalities, {result['duration']:.0f}s")

    # Look for video files
    print(f"\n{'-' * 50}")
    print("Extracting video frames...")
    video_frames = {}
    video_paths = {
        "webcam": os.path.join(DATA_DIR, "session_20260306_181143_webcam.avi"),
        "eye": r"C:\Users\chris\Downloads\neon_pov\eye_16-38.mp4",
        "pov": r"C:\Users\chris\Downloads\neon_pov\pov_16-38.mp4",
    }
    for vname, vpath in video_paths.items():
        frames = extract_video_frames(vpath)
        if frames:
            video_frames[vname] = frames
            print(f"  {vname}: {len(frames)} frames")
        else:
            print(f"  {vname}: not found")

    # Build aggregate statistics
    stats = {
        "totalSizeMB": round(total_size, 1),
        "totalDurationSec": round(total_duration, 1),
        "totalSessions": len(sessions),
        "totalSubjects": len(set(s["info"].get("subject", "?") for s in sessions)),
        "totalChannels": total_channels,
        "totalXdfFiles": len(xdf_files),
        "totalDataGB": round(sum(os.path.getsize(f) for f in xdf_files) / (1024**3), 1),
        "modalitiesSeen": sorted(list(all_modalities_seen)),
        "allFiles": [parse_session_info(f) for f in xdf_files],
    }

    # Build output
    output = {
        "version": 2,
        "stats": stats,
        "sessions": [],
        "videoFrames": video_frames,
    }

    for s in sessions:
        # Trim channel data to keep file size manageable
        session_out = {
            "info": s["info"],
            "sampleRate": s["data"]["sampleRate"],
            "duration": s["data"]["duration"],
            "nSamples": s["data"]["nSamples"],
            "streams": s["data"]["streams"],
            "modalities": {},
            "correlations": s["data"]["correlations"],
        }
        for mname, mdata in s["data"]["modalities"].items():
            # Keep mean signal, subsample channels (max 6 per modality)
            ch_data = mdata["channels"]
            n_ch = min(6, len(ch_data[0]) if ch_data else 0)
            ch_sub = [[row[i] for i in range(n_ch)] for row in ch_data] if n_ch > 0 else []
            session_out["modalities"][mname] = {
                "color": mdata["color"],
                "desc": mdata["desc"],
                "nChannels": mdata["nChannels"],
                "mean": mdata["mean"],
                "channels": ch_sub,
                "spectral": mdata.get("spectral"),
            }
        output["sessions"].append(session_out)

    # Write output
    out_path = os.path.join(OUT_DIR, "neurompedia_data.json")
    print(f"\n{'=' * 60}")
    print(f"Writing {out_path}...")
    with open(out_path, 'w') as f:
        json.dump(output, f)

    fsize = os.path.getsize(out_path) / (1024 * 1024)
    print(f"\nDone! {fsize:.1f} MB")
    print(f"\nAGGREGATE STATS:")
    print(f"  Total data on disk:  {stats['totalDataGB']} GB")
    print(f"  XDF files:           {stats['totalXdfFiles']}")
    print(f"  Sessions exported:   {stats['totalSessions']}")
    print(f"  Unique subjects:     {stats['totalSubjects']}")
    print(f"  Total duration:      {stats['totalDurationSec']:.0f}s ({stats['totalDurationSec']/60:.1f} min)")
    print(f"  Total channels:      {stats['totalChannels']}")
    print(f"  Modalities:          {', '.join(stats['modalitiesSeen'])}")

if __name__ == "__main__":
    main()
