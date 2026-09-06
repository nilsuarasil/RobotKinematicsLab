"""
TremorFilter'ın gerçekten işini yaptığını -- yüksek frekanslı (titreme)
bileşeni bastırırken düşük frekanslı (kasıtlı hareket) bileşeni koruduğunu --
sentetik bir sinyal (yavaş kasıtlı hareket + hızlı titreme) ile doğrular.
"""
import numpy as np
from src.control.tremor_filter import TremorFilter, DeadbandFilter, analyze_tremor_reduction


def make_synthetic_hand_signal(duration_s=4.0, sample_rate_hz=100.0,
                                intentional_hz=0.5, tremor_hz=10.0,
                                intentional_amp=1.0, tremor_amp=0.15, seed=0):
    """[N, 1] şeklinde: yavaş (istemli) sinüs + hızlı (titreme) sinüs + küçük
    beyaz gürültü. n_channels=1 kanal kullanıyoruz (tek eksende test yeterli)."""
    n = int(duration_s * sample_rate_hz)
    t = np.arange(n) / sample_rate_hz
    rng = np.random.default_rng(seed)

    intentional = intentional_amp * np.sin(2 * np.pi * intentional_hz * t)
    tremor = tremor_amp * np.sin(2 * np.pi * tremor_hz * t)
    noise = rng.normal(0, 0.02, size=n)

    signal = (intentional + tremor + noise).reshape(-1, 1)
    return signal, intentional.reshape(-1, 1)


def test_filter_significantly_reduces_tremor_band_energy():
    sample_rate = 100.0
    raw, _ = make_synthetic_hand_signal(sample_rate_hz=sample_rate)
    tf = TremorFilter(cutoff_hz=2.0, sample_rate_hz=sample_rate, n_channels=1)
    filtered = tf.filter_signal(raw)

    report = analyze_tremor_reduction(raw, filtered, sample_rate_hz=sample_rate)
    # Sentetik sinyale eklenen beyaz gürültü, titreme bandında da bir miktar
    # enerji taşıdığı için %100'e yakın bir azalma beklenmez -- yine de
    # ezici bir çoğunluk (>%70) süzülmeli.
    assert report["tremor_band_reduction_percent"] > 70.0


def test_filter_preserves_intentional_low_frequency_motion():
    """Filtrelenmiş sinyal, orijinal İSTEMLİ (yavaş) harekete -- ham sinyalden
    çok daha yakın kalmalı (genlik olarak büyük ölçüde korunmalı)."""
    sample_rate = 100.0
    raw, intentional = make_synthetic_hand_signal(sample_rate_hz=sample_rate)
    tf = TremorFilter(cutoff_hz=2.0, sample_rate_hz=sample_rate, n_channels=1)
    filtered = tf.filter_signal(raw)

    # Geçiş durumunun (transient) etkisini azaltmak için sinyalin ikinci
    # yarısında karşılaştır.
    half = len(raw) // 2
    intentional_amplitude = np.max(np.abs(intentional[half:]))
    filtered_amplitude = np.max(np.abs(filtered[half:]))
    assert filtered_amplitude > 0.7 * intentional_amplitude


def test_filter_rejects_invalid_cutoff_above_nyquist():
    import pytest
    with pytest.raises(ValueError):
        TremorFilter(cutoff_hz=30.0, sample_rate_hz=50.0)


def test_stateful_filter_sample_matches_batch_filter_signal():
    sample_rate = 50.0
    raw, _ = make_synthetic_hand_signal(sample_rate_hz=sample_rate, duration_s=2.0)

    tf_batch = TremorFilter(cutoff_hz=2.0, sample_rate_hz=sample_rate, n_channels=1)
    filtered_batch = tf_batch.filter_signal(raw)

    tf_stream = TremorFilter(cutoff_hz=2.0, sample_rate_hz=sample_rate, n_channels=1)
    filtered_stream = np.array([tf_stream.filter_sample(s) for s in raw])

    assert np.allclose(filtered_batch, filtered_stream, atol=1e-10)


def test_reset_clears_internal_state():
    """Sabit bir girdiyle (örn. hep 1.0) filtrenin durumu, başlangıç
    koşuluyla ('sosfilt_zi', tam olarak o sabit girdinin kararlı-durum
    tepkisi için tasarlanmıştır) çakışabilir -- bu yüzden durumu gerçekten
    hareket ettirmek için DEĞİŞEN bir girdi dizisi kullanıyoruz."""
    tf = TremorFilter(cutoff_hz=2.0, sample_rate_hz=50.0, n_channels=1)
    for value in [1.0, -1.0, 2.0, -2.0, 0.5]:
        tf.filter_sample([value])
    state_after_samples = tf._zi.copy()
    tf.reset()
    assert not np.allclose(state_after_samples, tf._zi)


def test_deadband_filter_zeros_small_motion():
    db = DeadbandFilter(threshold=0.5)
    assert np.allclose(db.filter_sample([0.1, 0.1, 0.0]), [0.0, 0.0, 0.0])


def test_deadband_filter_passes_large_motion_unchanged():
    db = DeadbandFilter(threshold=0.5)
    sample = np.array([1.0, 0.0, 0.0])
    assert np.allclose(db.filter_sample(sample), sample)
