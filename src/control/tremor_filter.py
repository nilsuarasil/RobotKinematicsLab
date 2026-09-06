"""
Surgical Tremor Filter: cerrahın elindeki fizyolojik titremeyi (genelde
8-12 Hz civarında), İSTEMLİ (kasıtlı) el hareketini (genelde < 2-3 Hz)
BOZMADAN bastırmak için gerçek zamanlı (streaming, "causal") bir alçak
geçiren (low-pass) filtre.

Neden low-pass? Titreme yüksek frekanslı, küçük genlikli bir salınımdır;
kasıtlı hareket ise düşük frekanslı, büyük genlikli bir sinyaldir. Kesim
frekansını (cutoff) ikisinin arasına (örn. 2 Hz) koyan bir Butterworth
filtresi, titremeyi büyük ölçüde söndürürken kasıtlı hareketi neredeyse
hiç geciktirmeden/bozmadan geçirir.

İki yöntem sunuyoruz:
  1. TremorFilter -- 2. derece Butterworth low-pass (asıl, "gerçek" filtre).
  2. DeadbandFilter -- çok küçük (titreme büyüklüğündeki) hareketleri
     tamamen sıfırlayan basit bir eşik filtresi (Butterworth'e tamamlayıcı,
     karşılaştırma için).

Her ikisi de STATEFUL: `filter_sample()` her çağrıda BİR YENİ örnek alır ve
filtrelenmiş değeri döndürür -- yani gerçek zamanlı bir teleoperation
döngüsünde her kontrol adımında bir kez çağrılmak üzere tasarlandı
(bkz. teleoperation/mouse_teleop.py).
"""
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi


class TremorFilter:
    """Çok kanallı (örn. 3 boyutlu Cartesian hız/konum), gerçek zamanlı
    (causal) Butterworth low-pass filtresi."""

    def __init__(self, cutoff_hz: float = 2.0, sample_rate_hz: float = 50.0,
                 order: int = 2, n_channels: int = 3):
        if cutoff_hz >= sample_rate_hz / 2:
            raise ValueError(
                f"cutoff_hz ({cutoff_hz}) Nyquist frekansından (sample_rate_hz/2 = "
                f"{sample_rate_hz / 2}) küçük olmalı."
            )
        self.cutoff_hz = cutoff_hz
        self.sample_rate_hz = sample_rate_hz
        self.n_channels = n_channels
        self.sos = butter(order, cutoff_hz, btype="low", fs=sample_rate_hz, output="sos")
        self._zi = self._initial_state()

    def _initial_state(self):
        """
        SIFIR (dinlenme) durumu: `sosfilt_zi(sos)` DOĞRUDAN kullanılırsa,
        bu aslında "girdi hep 1.0 sabitiymiş gibi" bir kararlı-durum başlangıcı
        verir (scipy'nin normal kullanımı bunu `zi * x[0]` ile ilk örneğe göre
        ölçeklemektir). Teleoperation'da master girdisi genelde SIFIRDAN
        (elin durgun olduğu bir andan) başladığı için doğru varsayım budur --
        `sosfilt_zi(sos) * 0 = 0` ile aynı şekle sahip sıfır matrisi kullanıyoruz.
        Aksi halde (yanlışlıkla) ~1.0 büyüklüğünde bir "hayalet" geçiş sinyali,
        küçük gerçek girdilerin (örn. piksel-ölçekli fare hareketi) üzerine
        binip filtrelenmiş çıktıyı yanlış büyütebilir.
        """
        zi_shape = sosfilt_zi(self.sos).shape  # (n_sections, 2)
        return np.zeros((self.n_channels,) + zi_shape)

    def reset(self):
        """İç filtre durumunu sıfırlar (yeni bir teleoperation oturumuna
        başlarken kullan -- aksi halde önceki oturumdan kalan geçiş durumu
        ilk birkaç örnekte küçük bir sıçramaya sebep olabilir)."""
        self._zi = self._initial_state()

    def filter_sample(self, sample):
        """sample: (n_channels,) -- tek bir zaman adımı. Filtrelenmiş
        (n_channels,) vektörü döndürür; iç durumu bir sonraki çağrı için saklar."""
        sample = np.asarray(sample, dtype=float)
        out = np.zeros(self.n_channels)
        for i in range(self.n_channels):
            y, self._zi[i] = sosfilt(self.sos, sample[i:i + 1], zi=self._zi[i])
            out[i] = y[0]
        return out

    def filter_signal(self, signal):
        """Offline/toplu filtreleme -- (N, n_channels) bir sinyali, dahili
        durumu SIFIRLAYIP baştan itibaren causal olarak filtreler (demo/test
        ve `analyze_tremor_reduction` için kullanılır)."""
        self.reset()
        signal = np.asarray(signal, dtype=float)
        out = np.zeros_like(signal)
        for t in range(signal.shape[0]):
            out[t] = self.filter_sample(signal[t])
        return out


class DeadbandFilter:
    """Büyüklüğü `threshold`'un altındaki hareketleri tamamen sıfırlayan
    basit bir eşik filtresi -- çok küçük (mikro-titreme büyüklüğündeki)
    girdileri tamamen görmezden gelir. Butterworth'ün aksine FAZ GECİKMESİ
    yoktur ama büyük genlikli titremeyi süzemez -- tamamlayıcı bir teknik."""

    def __init__(self, threshold: float):
        self.threshold = float(threshold)

    def filter_sample(self, sample):
        sample = np.asarray(sample, dtype=float)
        if np.linalg.norm(sample) < self.threshold:
            return np.zeros_like(sample)
        return sample


def generate_synthetic_hand_signal(duration_s: float = 4.0, sample_rate_hz: float = 100.0,
                                    intentional_hz: float = 0.5, tremor_hz: float = 10.0,
                                    intentional_amp: float = 1.0, tremor_amp: float = 0.15,
                                    n_channels: int = 3, seed: int = 0):
    """
    Demo/test amaçlı sentetik bir "cerrah eli" sinyali üretir: yavaş, kasıtlı
    bir hareket (düşük frekans) + fizyolojik titreme (yüksek frekans) + küçük
    ölçüm gürültüsü. (N, n_channels) şeklinde döndürür.
    """
    n = int(duration_s * sample_rate_hz)
    t = np.arange(n) / sample_rate_hz
    rng = np.random.default_rng(seed)

    intentional = intentional_amp * np.sin(2 * np.pi * intentional_hz * t)
    tremor = tremor_amp * np.sin(2 * np.pi * tremor_hz * t)
    noise = rng.normal(0, 0.02, size=n)
    base_signal = intentional + tremor + noise

    # Her kanala küçük bir faz kayması ekleyerek 3 kanalın birebir aynı
    # olmamasını sağlıyoruz (daha gerçekçi bir demo için).
    signal = np.stack([
        base_signal * np.cos(0.3 * ch) for ch in range(n_channels)
    ], axis=1)
    return signal


def format_tremor_demo_report(report: dict, cutoff_hz: float, sample_rate_hz: float) -> str:
    return "\n".join([
        "Tremor Filter Demo (synthetic surgeon's-hand signal):",
        f"Cutoff Frequency   = {cutoff_hz:.1f} Hz (sample rate: {sample_rate_hz:.0f} Hz)",
        f"Raw RMS            = {report['raw_rms']:.4f}",
        f"Filtered RMS       = {report['filtered_rms']:.4f}",
        f"Tremor Band Reduction (6-14 Hz) = {report['tremor_band_reduction_percent']:.1f}%",
    ])


def analyze_tremor_reduction(raw_signal, filtered_signal, sample_rate_hz: float,
                              tremor_band=(6.0, 14.0)):
    """
    Ham ve filtrelenmiş sinyalin FFT'sini karşılaştırarak titreme bandındaki
    (varsayılan 6-14 Hz -- fizyolojik tremor) enerjinin ne kadar azaldığını
    ölçer. Rapor/demo amaçlı (`--tremor-demo` bkz. main.py).

    Döndürür: dict {tremor_band_reduction_percent, raw_rms, filtered_rms}
    """
    raw_signal = np.asarray(raw_signal, dtype=float)
    filtered_signal = np.asarray(filtered_signal, dtype=float)
    n = raw_signal.shape[0]

    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate_hz)
    band_mask = (freqs >= tremor_band[0]) & (freqs <= tremor_band[1])

    def band_energy(sig):
        energy = 0.0
        for ch in range(sig.shape[1]):
            spectrum = np.abs(np.fft.rfft(sig[:, ch]))
            energy += float(np.sum(spectrum[band_mask] ** 2))
        return energy

    raw_energy = band_energy(raw_signal)
    filtered_energy = band_energy(filtered_signal)
    reduction_percent = 100.0 * (1.0 - filtered_energy / raw_energy) if raw_energy > 0 else 0.0

    return {
        "tremor_band_reduction_percent": reduction_percent,
        "raw_rms": float(np.sqrt(np.mean(raw_signal ** 2))),
        "filtered_rms": float(np.sqrt(np.mean(filtered_signal ** 2))),
    }
