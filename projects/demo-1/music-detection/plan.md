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