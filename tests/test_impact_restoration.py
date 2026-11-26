import numpy as np
from scipy.signal import butter, sosfilt

from clearsound.config import Config
from clearsound.core.gap_analyzer import GapAnalyzer
from clearsound.core.sound_matcher import SoundMatcher
from clearsound.core.sound_inserter import SoundInserter
from clearsound.library.sound_library import SoundLibrary


def run_test():
    sr = 44100
    dur = 5.0
    samples = int(sr * dur)
    t = np.linspace(0, dur, samples, endpoint=False)

    # 1. Voice stem: realistic speech simulation (vowel formants + natural modulation)
    voice = np.zeros(samples)
    mask = (t >= 1.0) & (t <= 3.5)
    t_v = t[mask]
    f0 = 140.0 + 15.0 * np.sin(2 * np.pi * 3.0 * t_v)  # natural pitch drift
    speech_tone = 0.4 * np.sin(2 * np.pi * f0 * t_v) + 0.25 * np.sin(2 * np.pi * 2 * f0 * t_v)
    speech_breath = 0.15 * np.random.randn(len(t_v))   # natural unvoiced consonants & breath
    voice[mask] = (speech_tone + speech_breath) * 0.4

    # 2. Explosion occurs at 1.8s
    st_exp = int(1.8 * sr)
    exp_samples = int(1.8 * sr)
    t_exp = np.linspace(0, 1.8, exp_samples, endpoint=False)
    exp_crack = np.exp(-t_exp * 50.0) * np.random.randn(exp_samples) * 0.9  # sharp percussive crack
    exp_sub = np.sin(2 * np.pi * 50 * t_exp) * np.exp(-t_exp * 2.0) * 0.8   # low rumble

    explosion = np.zeros(samples)
    explosion[st_exp:st_exp + exp_samples] = exp_crack + exp_sub

    original = voice + explosion

    # 3. Separation simulation:
    # Demucs puts voice + high-frequency percussive crack into vocals stem
    # and leaves only muffled sub-bass boom in no_vocals
    vocals = voice.copy()
    vocals[st_exp:st_exp + exp_samples] += exp_crack

    sos_muffle = butter(2, 400, btype="lowpass", fs=sr, output="sos")
    no_vocals = sosfilt(sos_muffle, explosion)

    config = Config()
    lib = SoundLibrary(config)
    lib.generate_basic_sounds()

    # 4. Gap Analyzer — returns 3-tuple now
    analyzer = GapAnalyzer(config)
    gaps, sound_events, speech_regions = analyzer.analyze(original, no_vocals, vocals, sr)
    print(f"Gaps found: {len(gaps)}, Sound events: {len(sound_events)}, Speech regions: {len(speech_regions)}")
    assert len(gaps) > 0, "No gaps detected!"
    g = gaps[0]
    print(f"Gap 0: event_type={g.event_type}, neural_category={g.neural_category}, category_hint={g.category_hint}")
    # Neural net should detect SOMETHING overlapping with the speech+explosion region
    has_non_speech = any(g2.has_overlapping_sounds for g2 in gaps)
    print(f"  At least one gap has overlapping sounds: {has_non_speech}")

    # 5. Sound Matcher
    matcher = SoundMatcher(config, lib)
    matches = matcher.find_matches(gaps)
    # Find gaps with matches (gaps that have overlapping sounds)
    matched_gaps = [i for i, m in matches.items() if m]
    print(f"Gaps with matches: {len(matched_gaps)}/{len(gaps)}")

    # 6. Sound Inserter — requires original for the new architecture
    inserter = SoundInserter(config)
    enhanced = inserter.insert(no_vocals, gaps, matches, sr, vocals=vocals, original=original)
    print(f"Enhanced shape: {enhanced.shape}, max: {np.max(np.abs(enhanced)):.2f}")

    # 7. Check that speech regions are silenced or reconstructed
    for g2 in gaps:
        st2, en2 = g2.start_sample, g2.end_sample
        seg_enh = enhanced[st2:en2]
        rms = float(np.sqrt(np.mean(seg_enh ** 2)))
        if g2.has_overlapping_sounds:
            print(f"  Gap with sounds [{g2.start_sec:.1f}-{g2.end_sec:.1f}]: RMS={rms:.4f} (should have content)")
        else:
            print(f"  Speech-only gap [{g2.start_sec:.1f}-{g2.end_sec:.1f}]: RMS={rms:.6f} (should be near-silent)")
            assert rms < 0.1, f"Speech-only gap should be near-silent, RMS={rms}"

    print(">>> ALL VERIFICATION TESTS PASSED SUCCESSFULLY! <<<")


if __name__ == "__main__":
    run_test()
