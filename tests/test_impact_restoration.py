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

    # 4. Gap Analyzer
    analyzer = GapAnalyzer(config)
    gaps = analyzer.analyze(original, no_vocals, vocals, sr)
    print(f"Gaps found: {len(gaps)}")
    assert len(gaps) > 0, "No gaps detected!"
    g = gaps[0]
    print(f"Gap 0: event_type={g.event_type}, category={g.category_hint}")
    assert g.category_hint == "impacts" or g.event_type in ["explosive_blast", "transient_impact"]

    # 5. Sound Matcher
    matcher = SoundMatcher(config, lib)
    matches = matcher.find_matches(gaps)
    top_match = matches[0][0]
    print(f"Top match: {top_match.sound.name} (category: {top_match.sound.category}, score: {top_match.similarity_score:.2f})")
    assert top_match.sound.category == "impacts"

    # 6. Sound Inserter with HPSS De-bleeding + Harmonic Exciter
    inserter = SoundInserter(config)
    enhanced = inserter.insert(no_vocals, gaps, matches, sr, vocals=vocals)
    print(f"Enhanced shape: {enhanced.shape}, max: {np.max(np.abs(enhanced)):.2f}")

    # 7. Check high frequency (> 3000 Hz) restoration
    sos_hi = butter(2, 3000, btype="highpass", fs=sr, output="sos")
    hi_novoc = np.sum(np.abs(sosfilt(sos_hi, no_vocals[st_exp:st_exp + int(0.5 * sr)])))
    hi_enh = np.sum(np.abs(sosfilt(sos_hi, enhanced[st_exp:st_exp + int(0.5 * sr)])))
    ratio = hi_enh / (hi_novoc + 1e-8)
    print(f"High frequency restoration factor: {ratio:.1f}x (from {hi_novoc:.3f} to {hi_enh:.3f})")
    assert ratio > 1.2, f"Expected high restoration factor > 1.2, got {ratio}"
    print(">>> ALL VERIFICATION TESTS PASSED SUCCESSFULLY! <<<")


if __name__ == "__main__":
    run_test()
