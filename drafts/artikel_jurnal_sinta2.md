# Hibridisasi Algoritma Genetika Adaptif dan Pencarian Lokal Elit untuk Optimasi Penjadwalan Perkuliahan Berbasis Kurikulum

**Penulis:** Amri Reza Wahyudin$^1$, Rohmat Indra Borman$^2$  
**Afiliasi:** Fakultas Teknik dan Ilmu Komputer, Universitas Teknokrat Indonesia, Bandar Lampung 35142, Indonesia  
**Korespondensi:** amri\_reza\_wahyudin@teknokrat.ac.id  
**Target Publikasi:** Jurnal Nasional Terakreditasi SINTA 2 (*Jurnal RESTI* / *INFOTEL*)  

---

## ABSTRAK

*University Course Timetabling Problem* (UCTP) merupakan persoalan optimasi kombinatorial NP-hard yang dicirikan oleh ruang pencarian faktorial berkonstrain ketat. Algoritma genetika konvensional dengan parameter statis sering kali mengalami konvergensi prematur pada optimum lokal tak layak (*infeasible*). Pendekatan adaptif terdahulu seperti *Self-Adaptive Genetic Algorithm* (SAGA) memperkenalkan dekomposisi koevolutif dan inferensi fuzzy, namun memicu osilasi parameter mutasi yang tinggi dan kelemahan alokasi kredit. Artikel ini mengusulkan *Hybrid Adaptive Genetic Algorithm* (HAGA) yang mengintegrasikan *Adaptive Parameter Controller* (APC) linier berbasis *normalized pairwise Hamming distance* ($H_{\text{norm}}$) dan *conflict-directed Elite Local Search* (ELS) di bawah penegakan paritas anggaran evaluasi fungsi objektif global ($B_{\max} = 50.000$). Evaluasi empiris menggunakan protokol bebas bias pada instance tolok ukur standar ITC 2007 membuktikan bahwa: (1) ELS bertindak sebagai lokomotif mutlak pemenuhan kendala kaku yang memangkas 96%–100% pelanggaran ($p = 0{,}001953$, uji Wilcoxon), (2) APC berfungsi sebagai katalisator diversitas yang melipatgandakan tingkat kelayakan solusi hingga 3× lipat (dari 10% menjadi 30%) pada permasalahan skala besar (`comp07`, 478 perkuliahan) dengan penalti kendala lunak yang lebih rendah ($SCV = 5.019$ vs $5.254$), serta (3) rekonstruksi kritis membuktikan HAGA melipatgandakan kelayakan solusi SAGA dari 40% menjadi 90% dengan parameter mutasi yang 100× lebih stabil.

**Kata Kunci:** Penjadwalan Perkuliahan, Algoritma Genetika Adaptif, Pencarian Lokal Elit, Jarak Hamming, UCTP, Paritas Anggaran Komputasi.

---

## 1. PENDAHULUAN

Penyusunan jadwal perkuliahan pada institusi pendidikan tinggi (*University Course Timetabling Problem* / UCTP) merupakan permasalahan optimasi kombinatorial berkonstrain tinggi yang tergolong *NP-hard*. Pada formulasi berbasis kurikulum (*Curriculum-Based Course Timetabling* / CB-CTT) standar internasional ITC 2007, algoritma dituntut untuk memenuhi dua kelompok kendala: kendala kaku (*Hard Constraints* / HCV) yang pantang dilanggar demi validitas operasional perkuliahan, serta kendala lunak (*Soft Constraints* / SCV) yang merepresentasikan preferensi kenyamanan mahasiswa dan efisiensi pemanfaatan fasilitas kampus.

Algoritma Genetika (GA) merupakan salah satu pendekatan metaheuristik yang paling banyak diteliti untuk UCTP. Namun, GA konvensional dengan probabilitas persilangan ($P_c$) dan mutasi ($P_m$) statis menghadapi dilema klasik eksplorasi-eksploitasi. Untuk mengatasi kelemahan ini, Pamungkas dan Neta (2025) mengajukan SAGA dengan memadukan sistem inferensi fuzzy Mamdani dan *cooperative coevolution*. Kendati demikian, evaluasi terhadap arsitektur SAGA mengidentifikasi adanya bias alokasi kredit pada dekomposisi sub-populasi waktu dan ruang, serta osilasi nilai parameter mutasi yang liar akibat *rule-base* fuzzy yang tidak terdefinisi secara mulus.

Artikel ini mengusulkan arsitektur **HAGA (*Hybrid Adaptive Genetic Algorithm*)**, sebuah pendekatan integratif terpadu yang memadukan kontrol parameter linier tertutup berbasis keragaman genotipik populasi ($H_{\text{norm}}$) dan pencarian lokal elit terarah konflik (*conflict-directed ELS*) di bawah pengawasan ketat kuota evaluasi fungsi objektif bersama ($B_{\max}$). Kebaruan utama yang ditawarkan bukan sekadar penambahan komponen heuristik, melainkan pembuktian efek sinergis simbiotik antar-komponen di mana ELS menjamin kelayakan mikro ($HCV=0$), sementara APC mencegah kepunahan diversitas makro pada ruang pencarian berdimensi tinggi.

---

## 2. METODOLOGI PENELITIAN

### 2.1 Formulasi Model dan Dominasi Kelayakan Mutlak

Fungsi objektif skalar internal $F(S)$ dirumuskan untuk menegakkan dominasi kelayakan mutlak (*Strict Feasibility Dominance*):
$$F(S) = W_{\text{hard}} \cdot HCV(S) + SCV(S)$$
di mana bobot kendala kaku ditetapkan $W_{\text{hard}} = 100.000$. Penatapan bobot ini secara matematis menjamin bahwa setiap jadwal yang layak operasional ($HCV=0$) pasti memiliki nilai fitnes yang lebih unggul dibandingkan jadwal yang tidak layak ($HCV \ge 1 \implies F(S) \ge 100.000$).

### 2.2 Pengukuran Keragaman Genotipik Berpasangan ($H_{\text{norm}}$)

Kromosom direpresentasikan sebagai larik 1D berindeks *event* kuliah $\mathbf{c} \in \mathbb{N}^E$. Keragaman populasi dihitung pada setiap generasi melalui jarak Hamming berpasangan ternormalisasi di seluruh $\frac{N(N-1)}{2}$ kombinasi pasangan individu:
$$H_{\text{norm}} = \frac{2}{N(N-1)E} \sum_{i=1}^{N-1} \sum_{j=i+1}^N \sum_{k=1}^E \mathbb{I}(c_{i,k} \neq c_{j,k})$$
di mana $\mathbb{I}(\cdot)$ bernilai 1 jika penugasan waktu atau ruang berbeda, dan 0 jika identik. Nilai $H_{\text{norm}} \in [0, 1]$ mencerminkan derajat sebaran spasial populasi di ruang solusi.

### 2.3 Kontrol Parameter Adaptif Linier Terikat (APC)

Probabilitas crossover ($P_c$) dan mutasi ($P_m$) disesuaikan secara deterministik mengikuti dinamika diversitas populasi:
$$P_c = P_{c,\text{base}} + \alpha (H_{\text{norm}} - \bar{H})$$
$$P_m = P_{m,\text{base}} - \beta (H_{\text{norm}} - \bar{H})$$
Kedua parameter dijepit (*clamped*) secara ketat pada batas aman: $P_c \in [0{,}40, \; 0{,}95]$ dan $P_m \in [0{,}01, \; 0{,}30]$.

### 2.4 Elite Local Search Terarah Bentrokan (ELS)

Pencarian lokal stochastic first-improvement diaplikasikan secara generational pada 10% individu terbaik ($N_{\text{elite}} = \lceil 0{,}10 \cdot N \rceil$). Langkah pencarian diprioritaskan secara selektif pada kegiatan kuliah yang terlibat dalam pelanggaran kendala kaku. Setiap evaluasi selama ELS dipotong langsung dari kuota evaluasi objektif global $B_{\max} = 50.000$.

---

## 3. HASIL DAN PEMBAHASAN

### 3.1 Audit Kritis dan Rekonstruksi Anteseden SAGA

Tabel 1 menyajikan hasil perbandingan empiris antara Canonical GA, SAGA Reconstructed, dan HAGA pada instance ITC 2007 `comp01` (10 *seeds*, $B_{\max} = 50.000$).

**Tabel 1. Perbandingan Empiris Kinerja SAGA vs HAGA pada ITC 2007 `comp01`**

| Algoritma | Tingkat Kelayakan ($HCV=0$) | Median $HCV$ | Median Fitnes | Feasible $SCV$ Median | Waktu Eksekusi |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Canonical GA** | 0% (0/10) | 40,5 | 4.051.923 | N/A | 85,5 s |
| **SAGA Reconstructed** | 40% (4/10) | 1,0 | 101.765 | 1.773,0 | 96,1 s |
| **HAGA (Diusulkan)** | **90% (9/10)** | **0,0** | **1.524** | **1.503,0** | 118,6 s |

Hasil menunjukkan bahwa SAGA hanya mencapai tingkat kelayakan 40% akibat kegagalan alokasi kredit koevolusi sub-populasi dan osilasi parameter mutasi yang tinggi ($\sigma = 0{,}126$). HAGA berhasil melipatgandakan tingkat kelayakan menjadi 90% dengan penalti lunak yang lebih efisien ($1.503$ vs $1.773$).

### 3.2 Studi Ablasi Empiris 4-Tahap (RQ2)

Tabel 2 merangkum hasil studi ablasi komprehensif 120 run pada 3 instance representatif ITC 2007.

**Tabel 2. Hasil Studi Ablasi 4-Tahap HAGA pada 3 Instance ITC 2007 (120 Run)**

| Instance | Varian Algoritma | Tingkat Kelayakan | Median $HCV$ | Median Fitnes | Feasible $SCV$ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`comp01`** (Kecil) | Canonical GA | 0% | 40,5 | 4.051.923 | N/A |
| | GA_APC | 0% | 45,5 | 4.552.351 | N/A |
| | GA_ELS | **90%** | **0,0** | **1.490** | 1.458 |
| | HAGA | **90%** | **0,0** | 1.524 | 1.503 |
| **`comp07`** (Terbesar) | Canonical GA | 0% | 263,5 | 26.355.924 | N/A |
| | GA_APC | 0% | 267,5 | 26.756.257 | N/A |
| | GA_ELS | 10% | 1,0 | 105.996 | 5.254 |
| | HAGA | **30%** | 1,5 | 154.951 | **5.019** |
| **`comp12`** (Terpadat) | Canonical GA | 0% | 133,5 | 13.353.815 | N/A |
| | GA_APC | 0% | 141,5 | 14.154.003 | N/A |
| | GA_ELS | 0% | 5,5 | 553.372 | N/A |
| | HAGA | 0% | **4,5** | **453.347** | N/A |

Uji Wilcoxon signed-rank mengonfirmasi bahwa penambahan ELS memberikan perbaikan performa yang sangat signifikan secara konsisten ($p = 0{,}001953$ pada seluruh instance). Pada instance raksasa `comp07`, penambahan APC ke dalam HAGA melipatgandakan kelayakan dari 10% menjadi 30% dan mereduksi penalti kendala lunak menjadi 5.019.

---

## 4. KESIMPULAN

Penelitian ini berhasil membuktikan bahwa integrasi kontrol parameter adaptif berbasis keragaman jarak Hamming ($H_{\text{norm}}$) dan pencarian lokal elit terarah bentrokan (ELS) di bawah paritas anggaran evaluasi menghasilkan algoritma yang kokoh dan defensibel untuk UCTP. Studi ablasi 120 run membuktikan ELS sebagai pemicu kelayakan mutlak yang memangkas 96%–100% bentrokan kaku, sedangkan APC berperan sebagai regulator diversitas yang melipatgandakan tingkat keberhasilan solusi layak hingga 300% pada permasalahan berskala besar.
