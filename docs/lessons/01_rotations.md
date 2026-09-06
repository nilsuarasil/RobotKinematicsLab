# Ders 1 — 3D Rotasyonlar (Rotation Matrices)

## Neden buradan başlıyoruz?

Bir robot kolunun her eklemi (joint), bir sonraki linki kendi ekseni etrafında
döndürür. 6-DOF bir robotu anlamak istiyorsak, önce "bir vektörü uzayda
döndürmek" ne demek, matematiksel olarak nasıl ifade edilir, onu çok sağlam
oturtmamız gerekiyor. Forward kinematics, DH parametreleri, Jacobian — hepsi
bu tek fikrin üzerine inşa ediliyor.

## 1. Bir noktayı döndürmek ne demek?

3B uzayda bir nokta (veya vektör) düşün: `p = [x, y, z]`. Bu noktayı, mesela
Z ekseni etrafında 90° döndürdüğümüzde, koordinatları değişir ama noktanın
orijine olan uzaklığı değişmez. Yani rotasyon, uzunlukları ve açıları koruyan
bir dönüşümdür (bir "izometri").

Bunu matematiksel olarak ifade etmenin yolu: `p' = R * p` — burada `R`, 3x3'lük
bir **rotasyon matrisi**, `p` ve `p'` ise 3x1'lik sütun vektörleri.

## 2. Temel eksen rotasyonları

Üç temel rotasyon matrisi var, her biri tek bir eksen etrafında döndürür.

**X ekseni etrafında θ kadar döndürme:**

$$
R_x(\theta) =
\begin{bmatrix}
1 & 0 & 0 \\
0 & \cos\theta & -\sin\theta \\
0 & \sin\theta & \cos\theta
\end{bmatrix}
$$

**Y ekseni etrafında θ kadar döndürme:**

$$
R_y(\theta) =
\begin{bmatrix}
\cos\theta & 0 & \sin\theta \\
0 & 1 & 0 \\
-\sin\theta & 0 & \cos\theta
\end{bmatrix}
$$

**Z ekseni etrafında θ kadar döndürme:**

$$
R_z(\theta) =
\begin{bmatrix}
\cos\theta & -\sin\theta & 0 \\
\sin\theta & \cos\theta & 0 \\
0 & 0 & 1
\end{bmatrix}
$$

### Bu matrisler nereden geliyor? (Rz örneği ile sezgi)

Z ekseni etrafında döndürürken, z koordinatı değişmez — sadece x-y düzlemindeki
nokta döner. Bunu 2B'de düşün: `(x, y)` noktası, θ kadar döndürüldüğünde:

```
x' = x cosθ - y sinθ
y' = x sinθ + y cosθ
z' = z
```

Bu iki denklemi matris formunda yazınca yukarıdaki `Rz(θ)` çıkıyor. `Rx` ve
`Ry` için de aynı mantık, sadece hangi eksenin "sabit" kaldığı değişiyor —
dikkat et: `Ry`'de işaretler diğerlerine göre ters gibi görünür
(`sinθ`'nın yeri değişir), bunun sebebi eksen sıralamasının (x→y→z→x
döngüsel) sağ-el kuralına uyması.

## 3. Rotasyon matrislerinin önemli özellikleri

Bir matrisin gerçekten bir rotasyon matrisi olması için şu üç özelliği
sağlaması gerekir:

1. **Ortogonaldir**: `R^T * R = I` (satır ve sütunları birim uzunlukta ve
   birbirine dik). Bu, `R^{-1} = R^T` demek — yani bir rotasyonu geri almak
   için matrisi ters çevirmek yerine transpozunu almak yeterli. Hesaplama
   açısından çok değerli bir kısayol.
2. **Determinantı +1'dir**: Determinantı -1 olan ortogonal matrisler de
   uzunlukları korur ama bunlar "yansıma" (reflection) içerir, gerçek bir
   rotasyon değildir.
3. **Kompozisyon**: İki rotasyonu art arda uygulamak, matrisleri çarpmak
   demektir: önce `R1` sonra `R2` uygularsan, toplam rotasyon `R2 * R1`'dir
   (soldan çarpılır, sona uygulanan solda yazılır). Matris çarpımı
   **komütatif değildir** — yani `R1 * R2 != R2 * R1` genelde. Bunu robotta
   şöyle düşün: önce X etrafında, sonra Z etrafında döndürmek; sırayı
   değiştirirsen robot farklı bir yöne bakar. Bu, robotikte eklem sırasının
   neden önemli olduğunun matematiksel kökeni.

## 4. Bunun robotla ilişkisi

UR5'in her eklemi, kendi lokal ekseni etrafında dönen bir "menteşe"dir. Her
eklemin döndürdüğü rotasyonu `R_i(q_i)` matrisiyle ifade edeceğiz (`q_i` o
eklemin açısı). Ders 3'te (DH parametreleri) bu rotasyonları, eklemler arası
öteleme (translation) ile birleştirip her linkin konumunu hesaplayacağız.
Ders 4'te ise tüm bu dönüşümleri zincirleyerek "eklem açıları verilince
end-effector nerede?" sorusunu cevaplayacağız (forward kinematics).

## Kod

`src/kinematics/rotations.py` içinde:

- `rot_x(theta)`, `rot_y(theta)`, `rot_z(theta)` — temel rotasyon matrisleri
- `is_rotation_matrix(R)` — bir matrisin geçerli bir rotasyon matrisi olup
  olmadığını (ortogonal + det=+1) kontrol eder

Testler (`tests/test_rotations.py`) bu üç özelliği (ortogonallik, determinant,
bilinen açılarda beklenen sonuç) doğruluyor.

## Kendine sorman gereken sorular

- `rot_z(0)` neden birim matris (identity) olmalı?
- `rot_x(90°)` uygulandığında, Y ekseni üzerindeki bir nokta nereye gider?
  (Kağıt üzerinde sağ el kuralıyla çizip kontrol et.)
- `rot_z(90°) @ rot_x(90°)` ile `rot_x(90°) @ rot_z(90°)` aynı sonucu verir mi?
  Kodda dene, neden farklı olduğunu düşün.
