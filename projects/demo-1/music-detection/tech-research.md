A practical starting point is **`librosa`**. It can estimate tempo from a short mono waveform using onset strength and beat tracking. For a five-second clip, treat the BPM as an estimate rather than ground truth: at 120 BPM, the clip contains only about 10 beats.

## Recommended approach

1. Convert the `AudioChunk` to a mono floating-point NumPy array.
2. Resample or assume a known sample rate.
3. Compute onset strength.
4. Estimate tempo with `librosa.beat.beat_track`.
5. Use the strength and number of detected beats as a rough confidence score.
6. Return `None` for BPM when the result is too unreliable.

Install:

```bash
pip install librosa soundfile
```

For the Reachy Mini Wireless robot, `librosa` may be relatively heavy. If CPU usage matters, consider `aubio` instead; it is designed for lightweight real-time onset and tempo tracking.

## Example detector

```python name=music_detector.py
import asyncio
from collections.abc import Callable

import librosa
import numpy as np
from pydantic import BaseModel


class DetectedMusicInfo(BaseModel):
    is_music: bool
    bpm: int | None = None
    confidence: float | None = None


class MusicDetector(_SoundDetector):
    def __init__(
        self,
        *,
        music_info_handler: Callable[[DetectedMusicInfo], None],
        end_on_detection: bool = False,
    ) -> None:
        super().__init__(
            delay_between_checks=5.0,
            end_on_detection=end_on_detection,
        )
        self._music_info_handler = music_info_handler

    async def _check_for_sound(self, audio_buffer: AudioChunk) -> bool:
        # Avoid blocking the asyncio event loop while librosa processes audio.
        music_info = await asyncio.to_thread(
            self._detect_music,
            audio_buffer,
        )

        self._music_info_handler(music_info)
        return music_info.is_music

    @staticmethod
    def _detect_music(audio_buffer: AudioChunk) -> DetectedMusicInfo:
        # Adapt this part to the actual AudioChunk API.
        samples = np.asarray(audio_buffer.samples, dtype=np.float32)
        sample_rate = audio_buffer.sample_rate

        if samples.ndim == 2:
            # Support either [channels, samples] or [samples, channels].
            if samples.shape[0] <= 8:
                samples = samples.mean(axis=0)
            else:
                samples = samples.mean(axis=1)

        samples = np.asarray(samples, dtype=np.float32)
        samples = samples - np.mean(samples)

        peak = np.max(np.abs(samples), initial=0.0)
        if peak > 1.0:
            samples = samples / peak

        rms = float(np.sqrt(np.mean(samples**2)))
        if rms < 0.005:
            return DetectedMusicInfo(
                is_music=False,
                confidence=0.0,
            )

        # Onset strength emphasizes rhythmic/transient activity.
        onset_env = librosa.onset.onset_strength(
            y=samples,
            sr=sample_rate,
            aggregate=np.median,
        )

        if len(onset_env) < 3 or float(np.max(onset_env)) <= 0.0:
            return DetectedMusicInfo(
                is_music=False,
                confidence=0.0,
            )

        tempo, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset_env,
            sr=sample_rate,
            start_bpm=120.0,
            bpm=librosa.beat.tempo_frequencies(
                len(onset_env),
                hop_length=512,
                sr=sample_rate,
            )[0]
            if False
            else 120.0,
            tightness=100,
        )

        # librosa has returned both scalars and one-element arrays in
        # different versions.
        bpm_value = float(np.asarray(tempo).reshape(-1)[0])
        beat_count = len(beat_frames)

        # This is a heuristic confidence, not a probability.
        onset_mean = float(np.mean(onset_env))
        onset_peak = float(np.max(onset_env))
        rhythmicity = onset_mean / (onset_peak + 1e-8)

        beat_score = min(1.0, beat_count / 6.0)
        energy_score = min(1.0, rms / 0.05)
        confidence = float(
            np.clip(
                0.55 * beat_score
                + 0.25 * energy_score
                + 0.20 * rhythmicity,
                0.0,
                1.0,
            )
        )

        valid_bpm = 40.0 <= bpm_value <= 240.0
        is_music = valid_bpm and beat_count >= 3 and confidence >= 0.45

        return DetectedMusicInfo(
            is_music=is_music,
            bpm=round(bpm_value) if is_music else None,
            confidence=confidence,
        )
```

The `bpm=` expression above can simply be replaced with `bpm=120.0`; the conditional form is not needed. A cleaner call is:

```python
tempo, beat_frames = librosa.beat.beat_track(
    onset_envelope=onset_env,
    sr=sample_rate,
    start_bpm=120.0,
    bpm=120.0,
    tightness=100,
)
```

However, depending on the installed `librosa` version, `bpm` may not be accepted by `beat_track`. The most portable version is therefore:

```python
tempo, beat_frames = librosa.beat.beat_track(
    onset_envelope=onset_env,
    sr=sample_rate,
    start_bpm=120.0,
    tightness=100,
)
```

## Important: BPM is not music classification

A speech recording can contain periodic syllables, and noise can create false onsets. BPM detection alone should not be used as a definitive music detector. For better classification, combine:

- RMS energy
- spectral flatness
- onset strength
- beat count
- estimated BPM stability
- optional speech/music classifier

For example, a simple additional check is spectral flatness:

```python
flatness = librosa.feature.spectral_flatness(y=samples)
mean_flatness = float(np.mean(flatness))

# Very high flatness often indicates noise.
# Very low flatness can indicate tonal or musical content.
not_obviously_noise = mean_flatness < 0.6
```

Then incorporate `not_obviously_noise` into the `is_music` decision.

## Better results from overlapping windows

Rather than analyzing exactly one five-second clip, use several overlapping clips:

```text
0.0–5.0 seconds
1.0–6.0 seconds
2.0–7.0 seconds
```

Estimate BPM for each and take the median:

```python
stable_bpm = int(round(np.median(bpm_estimates)))
```

This significantly reduces octave errors such as detecting 60 BPM instead of 120 BPM. You can also normalize estimates by considering common alternatives:

```python
candidates = [bpm / 2, bpm, bpm * 2]
```

and selecting the candidate closest to the previous estimate or expected range.

## Lightweight alternative: `aubio`

For on-robot processing, `aubio` is worth considering:

```bash
pip install aubio
```

It provides:

- onset detection
- tempo tracking
- beat detection
- lower CPU overhead than `librosa`

Aubio is generally better for continuous streaming, while `librosa` is easier for analyzing a complete five-second buffer.

## Reachy Mini-specific considerations

Because `_check_for_sound` is asynchronous, do not perform a large synchronous `librosa` computation directly in it. Using:

```python
await asyncio.to_thread(...)
```

prevents the detector from blocking other application tasks.

Also verify the actual `AudioChunk` fields. The example assumes:

```python
audio_buffer.samples
audio_buffer.sample_rate
```

but the repository search did not expose the detector or `AudioChunk` definitions in the indexed results, so those names may need to be adapted. The code-search results can be incomplete; more results are available in the [Reachy Mini code search UI](https://github.com/pollen-robotics/reachy_mini/search?q=AudioChunk&type=code).

Yes—**`aubio` is generally a better fit for on-robot Reachy Mini processing** than `librosa` because it is lightweight and designed for streaming onset/tempo detection. It should use less CPU and memory on the CM4, although exact install success depends on the robot’s Python version and ARM64 environment.

For an existing Reachy Mini virtual environment:

```bash
uv pip install aubio
```

If you manage dependencies in a `pyproject.toml`, prefer:

```bash
uv add aubio
```

Then verify:

```bash
python -c "import aubio; print(aubio.version)"
```

If the standard package has no compatible ARM64 wheel and compilation fails, try:

```bash
uv pip install aubio-ledfx
```

`aubio-ledfx` is a compatible fork that may provide prebuilt ARM64 wheels. Avoid `--system` unless you intentionally want to modify the system Python; the Reachy Mini app environment is preferable.

### Minimal aubio BPM example

```python
import aubio
import numpy as np


def estimate_bpm(
    samples: np.ndarray,
    sample_rate: int,
    *,
    hop_size: int = 512,
    buffer_size: int = 1024,
) -> float | None:
    samples = np.asarray(samples, dtype=np.float32)

    if samples.ndim == 2:
        samples = samples.mean(axis=1)

    samples = samples - np.mean(samples)

    peak = np.max(np.abs(samples), initial=0.0)
    if peak > 1.0:
        samples = samples / peak

    tempo_detector = aubio.tempo(
        "default",
        buffer_size,
        hop_size,
        sample_rate,
    )

    bpm_values: list[float] = []

    for start in range(0, len(samples), hop_size):
        frame = samples[start : start + hop_size]

        if len(frame) < hop_size:
            frame = np.pad(frame, (0, hop_size - len(frame)))

        frame = frame.astype(np.float32)
        is_beat = tempo_detector(frame)[0]

        if is_beat:
            bpm = float(tempo_detector.get_bpm())
            if 40.0 <= bpm <= 240.0:
                bpm_values.append(bpm)

    if not bpm_values:
        return None

    return float(np.median(bpm_values))
```

For a five-second clip, I would use aubio as a **candidate BPM detector**, then require at least several detected beats and possibly average across overlapping windows. `aubio` alone does not reliably determine whether audio is music versus speech or rhythmic noise.

### 1. uv Command to Install Python aubio Package

The current recommended command to install the aubio package using `uv` (Astral's fast Python packaging tool) is:

```
uv pip install aubio
```

or, for [project-managed installs](https://docs.astral.sh/uv/reference/uv/), use:

```
uv pip install --system aubio
```

`uv` is fully supported on Raspberry Pi OS aarch64 (64-bit), and works as a replacement for pip in nearly all Python package install workflows[[1]](https://pydevtools.com/handbook/how-to/how-to-run-python-scripts-on-a-raspberry-pi-with-uv/)[[2]](https://docs.astral.sh/uv/reference/policies/platforms/).

If you want the most recent features and best compatibility, consider the actively maintained fork, especially for Python 3.10+ and ARM:

```
uv pip install aubio-ledfx
```

(`aubio-ledfx` is a drop-in replacement with prebuilt wheels for Linux ARM64.)[[3]](https://pypi.org/project/aubio-ledfx/)

---

### 2. ARM64 (Raspberry Pi CM4) Support

**Yes, aubio supports Linux ARM64, including the Raspberry Pi CM4.**  
- The mainline aubio project is periodically packaged for ARM64 Linux distros such as Arch Linux ARM[[4]](https://archlinuxarm.org/packages/aarch64/python-aubio).
- The `aubio-ledfx` fork provides prebuilt wheels for Python 3.8–3.14 on Linux ARM64 (manylinux), which means you can use `pip` or `uv` to install it directly on Raspberry Pi CM4 running 64-bit Raspberry Pi OS[[3]](https://pypi.org/project/aubio-ledfx/).

> **Summary:**  
> - For the latest, easiest ARM64 install, use: `uv pip install aubio-ledfx`
> - For the classic package: `uv pip install aubio`
> - ARM64 (Raspberry Pi 4/CM4, 64-bit OS) is well supported by both `uv` and the aubio Python package.

Let me know if you want platform-specific troubleshooting or build-from-source instructions!

---

1. [How to run Python scripts on a Raspberry Pi with uv](https://pydevtools.com/handbook/how-to/how-to-run-python-scripts-on-a-raspberry-pi-with-uv/)
2. [Platform support | uv - Astral](https://docs.astral.sh/uv/reference/policies/platforms/)
3. [aubio-ledfx · PyPI](https://pypi.org/project/aubio-ledfx/)
4. [python-aubio (aarch64) | Packages | Arch Linux ARM](https://archlinuxarm.org/packages/aarch64/python-aubio)

Yes—**both are possible**.

## Option 1: Process on the Reachy Mini robot

For a **Reachy Mini Wireless**, this is the best option for autonomous behavior:

```text
Reachy Mini microphone
        ↓
Reachy Mini SDK / GStreamer
        ↓
aubio BPM detection
        ↓
robot movement or reaction
```

Your Python app runs on the robot, captures audio with:

```python
mini.media.start_recording()
sample = mini.media.get_audio_sample()
```

and passes the samples directly to `aubio`. The SDK exposes the microphone sample rate through:

```python
sample_rate = mini.media.get_input_audio_samplerate()
```

The repository’s recording example follows this pattern: [examples/sound_record.py](https://github.com/pollen-robotics/reachy_mini/blob/main/examples/sound_record.py).

Install `aubio` in the robot/app environment:

```bash
uv pip install aubio
```

For an app managed by a `pyproject.toml`:

```bash
uv add aubio
```

This avoids sending approximately five seconds of raw audio over the network and is preferable for continuous music-reactive behavior.

## Option 2: Process on your MacBook

You can also run the detector on your Mac.

### MacBook microphone

If you want to analyze audio captured by the Mac itself, use a local audio input library such as `sounddevice` or the SDK’s local media backend:

```bash
uv add aubio sounddevice
```

### Reachy Mini microphone streamed to the Mac

If the audio should come from Reachy Mini’s microphone array, connect using the SDK’s WebRTC media backend:

```python
from reachy_mini import ReachyMini

with ReachyMini(media_backend="webrtc") as mini:
    mini.media.start_recording()

    sample_rate = mini.media.get_input_audio_samplerate()
    sample = mini.media.get_audio_sample()
```

The SDK supports `default`, `local`, and `webrtc` media backends in its recording example. The WebRTC option lets your Mac receive the robot’s audio and run `aubio` locally.

## Which should you choose?

| Requirement | Recommended location |
|---|---|
| Robot should react autonomously | Robot |
| Lowest audio latency | Robot |
| Lowest CM4 CPU usage | MacBook |
| Easier development and debugging | MacBook |
| Testing with recorded clips | MacBook |
| Published/on-robot Python app | Robot |
| Remote web application | MacBook/browser side |

For your `MusicDetector`, I would use:

```text
Wireless Reachy Mini:
    media samples → aubio on robot → DetectedMusicInfo → robot motion

MacBook development:
    WebRTC samples → aubio on Mac → tune detector
```

Then move the same detector code to the robot once the thresholds are working.

## Important backend distinction

- **Reachy Mini Wireless:** the microphone physically belongs to the robot. For on-robot processing, use the local/GStreamer path.
- **Reachy Mini Lite:** the robot is controlled by your computer, so audio is generally available through the computer’s local audio system.
- **MacBook controlling Wireless Reachy Mini:** use `media_backend="webrtc"` if detection runs on the Mac; use the robot-side app/local backend if detection runs on the robot.

A minimal robot-side collection loop would look like this:

```python name=robot_music_detector.py
import time
from collections import deque

import aubio
import numpy as np
from reachy_mini import ReachyMini


HOP_SIZE = 512
BUFFER_SIZE = 1024
WINDOW_SECONDS = 5.0


def estimate_bpm(samples: np.ndarray, sample_rate: int) -> float | None:
    samples = np.asarray(samples, dtype=np.float32)

    if samples.ndim == 2:
        samples = samples.mean(axis=1)

    samples -= np.mean(samples)

    peak = np.max(np.abs(samples), initial=0.0)
    if peak > 1.0:
        samples /= peak

    tempo_detector = aubio.tempo(
        "default",
        BUFFER_SIZE,
        HOP_SIZE,
        sample_rate,
    )

    bpm_values = []

    for start in range(0, len(samples), HOP_SIZE):
        frame = samples[start : start + HOP_SIZE]

        if len(frame) < HOP_SIZE:
            frame = np.pad(frame, (0, HOP_SIZE - len(frame)))

        if tempo_detector(frame)[0]:
            bpm = float(tempo_detector.get_bpm())
            if 40 <= bpm <= 240:
                bpm_values.append(bpm)

    if not bpm_values:
        return None

    return float(np.median(bpm_values))


with ReachyMini(media_backend="default") as mini:
    mini.media.start_recording()

    sample_rate = mini.media.get_input_audio_samplerate()
    required_samples = int(WINDOW_SECONDS * sample_rate)
    audio_window = deque(maxlen=required_samples)

    try:
        while True:
            sample = mini.media.get_audio_sample()

            if sample is None:
                time.sleep(0.005)
                continue

            audio_window.extend(np.asarray(sample).reshape(-1))

            if len(audio_window) >= required_samples:
                bpm = estimate_bpm(
                    np.asarray(audio_window),
                    sample_rate,
                )

                print(f"Estimated BPM: {bpm}")
    finally:
        mini.media.stop_recording()
```

For the Wireless robot, use an app configured without unnecessary video streaming to reduce CPU load—for example, the repository’s app template mentions `gstreamer_no_video` for apps that do not use video.

The repository search results may be incomplete; you can inspect additional audio-related code in the [Reachy Mini code search UI](https://github.com/pollen-robotics/reachy_mini/search?q=get_audio_sample&type=code).

----

Yes. For **Reachy Mini Lite**, the computation can happen entirely on your MacBook while the SDK controls the robot.

```text
MacBook microphone / Reachy Mini Lite microphone
        ↓
Reachy Mini SDK media backend
        ↓
aubio on macOS
        ↓
DetectedMusicInfo
        ↓
Reachy Mini movement or behavior
```

Install `aubio` in the same environment as your Reachy Mini SDK:

```bash
uv add aubio numpy
```

If you are not using a `pyproject.toml`:

```bash
uv pip install aubio numpy
```

For Lite, use the local/default media backend rather than `webrtc`:

```python
from reachy_mini import ReachyMini

with ReachyMini(media_backend="default") as mini:
    mini.media.start_recording()

    sample_rate = mini.media.get_input_audio_samplerate()
    sample = mini.media.get_audio_sample()

    mini.media.stop_recording()
```

The SDK’s local audio implementation uses macOS audio through GStreamer’s `osxaudiosrc`; the `default` backend is therefore appropriate when the daemon and application are running on your Mac. The SDK recording example uses the same `start_recording()` / `get_audio_sample()` flow: [examples/sound_record.py](https://github.com/pollen-robotics/reachy_mini/blob/main/examples/sound_record.py).

A five-second rolling detector could look like this:

```python name=music_detector_lite.py
import time
from collections import deque

import aubio
import numpy as np
from reachy_mini import ReachyMini


WINDOW_SECONDS = 5.0
HOP_SIZE = 512
BUFFER_SIZE = 1024


def estimate_bpm(samples: np.ndarray, sample_rate: int) -> float | None:
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    samples -= np.mean(samples)

    peak = np.max(np.abs(samples), initial=0.0)
    if peak > 1.0:
        samples /= peak

    tempo = aubio.tempo(
        "default",
        BUFFER_SIZE,
        HOP_SIZE,
        sample_rate,
    )

    estimates: list[float] = []

    for offset in range(0, len(samples), HOP_SIZE):
        frame = samples[offset : offset + HOP_SIZE]

        if len(frame) < HOP_SIZE:
            frame = np.pad(frame, (0, HOP_SIZE - len(frame)))

        if tempo(frame)[0]:
            bpm = float(tempo.get_bpm())

            if 40.0 <= bpm <= 240.0:
                estimates.append(bpm)

    if not estimates:
        return None

    # Median is more robust than using the final aubio estimate.
    return float(np.median(estimates))


with ReachyMini(media_backend="default") as mini:
    mini.media.start_recording()

    sample_rate = mini.media.get_input_audio_samplerate()
    window_size = int(WINDOW_SECONDS * sample_rate)
    audio_window: deque[float] = deque(maxlen=window_size)

    print(f"Listening at {sample_rate} Hz...")

    try:
        while True:
            sample = mini.media.get_audio_sample()

            if sample is None:
                time.sleep(0.005)
                continue

            audio_window.extend(
                np.asarray(sample, dtype=np.float32).reshape(-1)
            )

            if len(audio_window) >= window_size:
                bpm = estimate_bpm(
                    np.asarray(audio_window),
                    sample_rate,
                )
                print(f"BPM: {bpm}")
    finally:
        mini.media.stop_recording()
```

For your `MusicDetector`, the important adaptation is that `AudioChunk` should contain or wrap the arrays returned by:

```python
mini.media.get_audio_sample()
```

and the sample rate should come from:

```python
mini.media.get_input_audio_samplerate()
```

You do **not** need to install `aubio` on the robot for the Lite setup. You also do not need `media_backend="webrtc"` when the daemon and your Python app are both running locally on the Mac.