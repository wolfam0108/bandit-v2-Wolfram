"""
Memory-efficient inference for Bandit-v2
Processes chunks one at a time to minimize VRAM usage
"""

import os
import sys
import time
import torch
import torchaudio as ta
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.bandit.bandit import Bandit


def create_model(stems, fs=48000):
    """Create Bandit model"""
    model = Bandit(
        in_channels=1,  # Model trained on mono, but uses treat_channel_as_feature
        stems=stems,
        band_type="musical",
        n_bands=64,
        normalize_channel_independently=False,
        treat_channel_as_feature=True,
        n_sqm_modules=8,
        emb_dim=128,
        rnn_dim=256,
        bidirectional=True,
        rnn_type="GRU",
        mlp_dim=512,
        hidden_activation="Tanh",
        hidden_activation_kwargs=None,
        complex_mask=True,
        use_freq_weights=True,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        window_fn="hann_window",
        wkwargs=None,
        power=None,
        center=True,
        normalized=True,
        pad_mode="reflect",
        onesided=True,
        fs=fs,
    )
    return model


def simple_chunked_inference(
    model, 
    audio,  # (channels, samples)
    fs=48000,
    chunk_seconds=30.0,  # Much larger chunks, less overlap
    overlap_seconds=2.0,  # Small overlap for crossfade
    device="cuda",
    use_half=True,
):
    """
    Simple chunk-based inference with crossfade overlap
    Processes ONE chunk at a time to minimize VRAM
    """
    n_channels, n_samples = audio.shape
    chunk_samples = int(chunk_seconds * fs)
    overlap_samples = int(overlap_seconds * fs)
    hop_samples = chunk_samples - overlap_samples
    
    # Create output buffers (on CPU to save VRAM)
    stems = model.stems
    outputs = {stem: torch.zeros(n_channels, n_samples) for stem in stems}
    
    # Create crossfade window
    fade_in = torch.linspace(0, 1, overlap_samples)
    fade_out = torch.linspace(1, 0, overlap_samples)
    
    # Calculate chunks
    n_chunks = max(1, (n_samples - overlap_samples) // hop_samples + 1)
    
    print(f"Processing {n_chunks} chunks of {chunk_seconds}s with {overlap_seconds}s overlap")
    
    for i in tqdm(range(n_chunks), desc="Processing"):
        start = i * hop_samples
        end = min(start + chunk_samples, n_samples)
        
        # Get chunk
        chunk = audio[:, start:end]
        actual_len = chunk.shape[1]
        
        # Pad if needed
        if actual_len < chunk_samples:
            chunk = torch.nn.functional.pad(chunk, (0, chunk_samples - actual_len))
        
        # Process on GPU
        chunk_gpu = chunk[None, :, :].to(device)
        
        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=use_half):
                batch = {"mixture": {"audio": chunk_gpu}}
                result = model(batch)
        
        # Get results back to CPU immediately
        for stem in stems:
            stem_audio = result["estimates"][stem]["audio"][0, :, :actual_len].float().cpu()
            
            # Apply crossfade for overlap regions
            if i == 0:
                # First chunk - no fade in
                outputs[stem][:, start:start+actual_len] = stem_audio
            else:
                # Apply crossfade in overlap region
                overlap_start = start
                overlap_end = start + overlap_samples
                
                if overlap_end <= n_samples:
                    # Fade out previous, fade in current
                    outputs[stem][:, overlap_start:overlap_end] *= fade_out
                    outputs[stem][:, overlap_start:overlap_end] += stem_audio[:, :overlap_samples] * fade_in
                    
                    # Copy rest without fade
                    if overlap_samples < actual_len:
                        outputs[stem][:, overlap_end:start+actual_len] = stem_audio[:, overlap_samples:actual_len]
                else:
                    outputs[stem][:, start:start+actual_len] = stem_audio
        
        # Clear GPU cache
        del chunk_gpu, result
        torch.cuda.empty_cache()
    
    return outputs


def run_inference(
    checkpoint_path: str,
    audio_path: str,
    output_dir: str = None,
    stems: list = None,
    fs: int = 48000,
    device: str = "cuda",
    use_half: bool = True,
    chunk_seconds: float = 30.0,
    overlap_seconds: float = 2.0,
):
    if stems is None:
        stems = ["speech", "music", "sfx"]
    
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(audio_path), "estimates")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print(f"\n=== Config ===")
    print(f"Chunk: {chunk_seconds}s, Overlap: {overlap_seconds}s")
    print(f"Half precision: {use_half}")
    
    # Load model
    print(f"\n=== Loading Model ===")
    model = create_model(stems=stems, fs=fs)
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    
    cleaned = {k.replace("model.", "", 1) if k.startswith("model.") else k: v 
               for k, v in state_dict.items()}
    
    model.load_state_dict(cleaned, strict=False)
    model.to(device)
    # Note: Keep model in FP32, autocast handles mixed precision during inference
    model.eval()
    print("Model loaded")
    
    # Load audio
    print(f"\n=== Loading Audio ===")
    audio, audio_fs = ta.load(audio_path)
    duration = audio.shape[1] / audio_fs
    print(f"Duration: {duration:.1f}s, Channels: {audio.shape[0]}, SR: {audio_fs}")
    
    if audio_fs != fs:
        print(f"Resampling {audio_fs} -> {fs}")
        audio = ta.functional.resample(audio, audio_fs, fs)
    
    n_channels = audio.shape[0]
    
    # Run inference
    print(f"\n=== Inference ===")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    
    # Process each channel separately to preserve stereo
    all_outputs = []
    for ch in range(n_channels):
        if n_channels > 1:
            print(f"\nProcessing channel {ch+1}/{n_channels}...")
        
        audio_ch = audio[ch:ch+1, :]  # Keep dim: (1, samples)
        
        ch_outputs = simple_chunked_inference(
            model, audio_ch, fs=fs,
            chunk_seconds=chunk_seconds,
            overlap_seconds=overlap_seconds,
            device=device,
            use_half=use_half,
        )
        all_outputs.append(ch_outputs)
    
    # Combine channels
    outputs = {}
    for stem in model.stems:
        outputs[stem] = torch.cat([ch_out[stem] for ch_out in all_outputs], dim=0)
    
    elapsed = time.time() - t0
    peak_vram = torch.cuda.max_memory_allocated() / 1024**3
    
    print(f"\nDone in {elapsed:.1f}s ({duration/elapsed:.1f}x realtime)")
    print(f"Peak VRAM: {peak_vram:.2f} GB")
    
    # Save
    print(f"\n=== Saving ===")
    for stem, audio_out in outputs.items():
        path = os.path.join(output_dir, f"{stem}_estimate.wav")
        ta.save(path, audio_out, fs)
        print(f"  {path}")
    
    print("\nComplete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-c", required=True)
    parser.add_argument("--audio", "-a", required=True)
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--stems", nargs="+", default=["speech", "music", "sfx"])
    parser.add_argument("--fs", type=int, default=48000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-half", action="store_true")
    parser.add_argument("--chunk", type=float, default=30.0, help="Chunk size in seconds")
    parser.add_argument("--overlap", type=float, default=2.0, help="Overlap in seconds")
    
    args = parser.parse_args()
    
    run_inference(
        checkpoint_path=args.checkpoint,
        audio_path=args.audio,
        output_dir=args.output,
        stems=args.stems,
        fs=args.fs,
        device=args.device,
        use_half=not args.no_half,
        chunk_seconds=args.chunk,
        overlap_seconds=args.overlap,
    )
