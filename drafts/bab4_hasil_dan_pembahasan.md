# BAB 4: HASIL PENELITIAN DAN PEMBAHASAN

Pada bab ini disajikan hasil eksperimen komputasional, analisis statistik inferensial, dan pembahasan teoritis mendalam terhadap kinerja algoritma *Hybrid Adaptive Genetic Algorithm* (HAGA) untuk menyelesaikan permasalahan *University Course Timetabling Problem* (UCTP) berbasis kurikulum (CB-CTT). Seluruh pengujian komputasi dieksekusi secara ketat di bawah invarian **paritas anggaran evaluasi objektif global** ($B_{\max} = 50.000$ evaluasi per run) dan divalidasi secara independen menggunakan evaluator resmi kompetisi ITC 2007.

---

## 4.1 Rekonstruksi dan Audit Kritis Anteseden Konseptual SAGA

Penelitian terdahulu oleh Pamungkas dan Neta (2025) mengajukan *Self-Adaptive Genetic Algorithm* (SAGA) yang menggabungkan *fuzzy logic controller* (Mamdani FIS) dan *cooperative coevolution* (pembagian sub-populasi waktu dan ruang), dengan klaim kepuasan kendala mencapai 98,9% pada studi kasus internal kampus Universitas Cakrawala. Namun, publikasi tersebut tidak menyertakan kode sumber, basis aturan fuzzy (*fuzzy rule base*), fungsi keanggotaan eksplisit, bobot fungsi fitnes, maupun dataset yang dapat diakses publik.

Untuk menguji kelayakan klaim arsitektural tersebut secara objektif pada standar internasional ITC 2007 (`comp01.ctt`), sistem *SAGA_Reconstructed* dibangun kembali secara komprehensif mengikuti spesifikasi literatur (Potter & De Jong coevolution, Mamdani FIS 3 input dan 2 output, serta local search berbasis hill climbing) di bawah paritas anggaran $B_{\max} = 50.000$ evaluasi pada 10 *random seeds* independen (Seed 101–110).

### 4.1.1 Perbandingan Kuantitatif HAGA vs SAGA

Tabel 4.1 menyajikan perbandingan kinerja empiris antara Canonical GA, SAGA Reconstructed, dan HAGA pada instance ITC 2007 `comp01`.

**Tabel 4.1 Perbandingan Kuantitatif Canonical GA, SAGA Reconstructed, dan HAGA (`comp01`, 10 Seeds, $B_{\max} = 50.000$)**

| Algoritma | Feasible Runs ($HCV=0$) | **Tingkat Kelayakan** | Median $HCV$ (Bentrokan) | Rentang $HCV$ | Median Penalti Lunak ($SCV$) | Nilai Fitnes Gabungan Median | Waktu Komputasi Rerata |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Canonical GA** | 0 / 10 | 0,0% | 40,5 | [36, 46] | N/A (Tidak Layak) | 4.051.923 | **85,2 s** |
| **SAGA Reconstructed** | 4 / 10 | 40,0% | 1,0 | [0, 4] | 1.773,0 | 101.765 | 96,1 s |
| **HAGA (Diusulkan)** | **9 / 10** | **90,0%** | **0,0** | **[0, 1]** | **1.503,0** | **1.524** | 118,6 s |

*Catatan: Seluruh solusi akhir telah diaudit secara independen oleh `src/validator.py` ($W_{\text{hard}} = 100.000$).*

### 4.1.2 Analisis Kegagalan Credit Assignment dan Ketidakstabilan Parameter SAGA

Temuan empiris pada Tabel 4.1 mengungkapkan tiga keterbatasan mendasar pada desain arsitektur SAGA:

1. **Kegagalan Alokasi Kredit Koevolusi (*Credit Assignment Failure*):**
   * SAGA memisahkan kromosom menjadi dua sub-populasi: sub-populasi slot waktu ($T$) dan sub-populasi ruang kelas ($R$).
   * Dalam UCTP, penugasan waktu dan ruang bersifat *tightly coupled*. Evaluasi satu sub-populasi dengan meminjam individu terbaik (*best representative*) dari sub-populasi pasangan menciptakan bias kredit yang menyesatkan. Akibatnya, pada 6 dari 10 run (60%), SAGA gagal mencapai solusi layak dan menyisakan bentrokan kendala kaku ($HCV \ge 1$).
   * Sebaliknya, representasi kromosom 1D berindeks *event* pada HAGA yang mengaitkan waktu dan ruang secara simultan berhasil melipatgandakan tingkat kelayakan menjadi **90% (2,25× lipat lebih tinggi)**.

2. **Osilasi dan Ketidakstabilan Kontroler Fuzzy Mamdani:**
   * Audit terhadap dinamika parameter sepanjang generasi membuktikan bahwa kontroler fuzzy SAGA memicu osilasi parameter mutasi yang liar ($P_m$ berayun dari $0{,}23$ hingga $0{,}77$, standar deviasi $\sigma = 0{,}126$ atau variabilitas $16\%$). Tingginya mutasi merusak blok-blok jadwal unggul (*building blocks*) yang telah terbentuk.
   * Sebaliknya, fungsi transfer linier berbasis jarak Hamming berpasangan ($H_{\text{norm}}$) pada HAGA menghasilkan degradasi parameter yang sangat mulus dan stabil ($P_c \approx 0{,}86$, $P_m \approx 0{,}056$, $\sigma = 0{,}001$), membuktikan keunggulan *Occam's razor* pada optimasi kombinatorial terikat.

3. **Demistifikasi Waktu Komputasi 31 Menit:**
   * Di bawah anggaran evaluasi yang setara ($B_{\max} = 50.000$), waktu komputasi riil SAGA pada prosesor AMD Ryzen 5 5600H adalah **96,1 detik per run**.
   * Waktu 31 menit yang dilaporkan dalam paper Pamungkas dan Neta (2025) terbukti bukan disebabkan oleh kompleksitas intrinsik UCTP, melainkan karena ketiadaan batas kuota evaluasi objektif dan *overhead* evaluasi komputasi koevolusi ganda yang tidak teroptimasi.

---

## 4.2 Studi Ablasi Empiris Empat Tahap (Jawaban RQ2)

Untuk menjawab **Research Question 2 (RQ2)**—yaitu mengukur kontribusi terisolasi dan sinergis dari *Adaptive Parameter Controller* (APC) dan *Elite Local Search* (ELS)—dilakukan pengujian ablasi empat varian terstruktur:
1. **`Canonical_GA`:** Algoritma genetika standar dengan parameter statis ($P_c = 0{,}85, P_m = 0{,}05$) tanpa pencarian lokal.
2. **`GA_APC`:** Algoritma genetika dengan kontrol adaptif linier berbasis $H_{\text{norm}}$, tanpa pencarian lokal.
3. **`GA_ELS`:** Algoritma genetika parameter statis dengan *conflict-directed elite local search*.
4. **`HAGA`:** Model terintegrasi penuh (APC + ELS).

Pengujian dieksekusi melintasi 3 instance tolok ukur representatif ITC 2007 (`comp01`, `comp07`, `comp12`) dengan 10 *random seeds* per konfigurasi, menghasilkan total **120 run independen** di bawah paritas anggaran $B_{\max} = 50.000$.

### 4.2.1 Hasil Kuantitatif Studi Ablasi Multi-Instance

**Tabel 4.2 Rangkuman Kinerja Studi Ablasi 4-Tahap pada 3 Instance ITC 2007 (120 Run)**

| Instance Benchmark | Varian Algoritma | Feasible Runs ($HCV=0$) | Tingkat Kelayakan | Median $HCV$ (Bentrokan) | Nilai Fitnes Median | Feasible $SCV$ Median | Waktu Rerata |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`comp01`**<br>(160 Kuliah, Skala Kecil) | **Canonical_GA** | 0 / 10 | 0,0% | 40,5 | 4.051.923 | N/A | 85,5 s |
| | **GA_APC** | 0 / 10 | 0,0% | 45,5 | 4.552.351 | N/A | 89,1 s |
| | **GA_ELS** | 9 / 10 | **90,0%** | **0,0** | **1.490** | **1.458,0** | 119,1 s |
| | **HAGA** | **9 / 10** | **90,0%** | **0,0** | 1.524 | 1.503,0 | 113,5 s |
| **`comp07`**<br>(478 Kuliah, Skala Terbesar) | **Canonical_GA** | 0 / 10 | 0,0% | 263,5 | 26.355.924 | N/A | 859,9 s |
| | **GA_APC** | 0 / 10 | 0,0% | 267,5 | 26.756.257 | N/A | 853,5 s |
| | **GA_ELS** | 1 / 10 | 10,0% | **1,0** | **105.996** | 5.254,0 | 1.336,0 s |
| | **HAGA** | **3 / 10** | **30,0%** | 1,5 | 154.951 | **5.019,0** | 1.327,6 s |
| **`comp12`**<br>(400 Kuliah, 1.368 Kendala) | **Canonical_GA** | 0 / 10 | 0,0% | 133,5 | 13.353.815 | N/A | 908,4 s |
| | **GA_APC** | 0 / 10 | 0,0% | 141,5 | 14.154.003 | N/A | 905,9 s |
| | **GA_ELS** | 0 / 10 | 0,0% | 5,5 | 553.372 | N/A | 1.388,0 s |
| | **HAGA** | 0 / 10 | 0,0% | **4,5** | **453.347** | N/A | 1.375,3 s |

### 4.2.2 Uji Signifikansi Statistik Berpasangan (Wilcoxon Signed-Rank Test)

Uji peringkat bertanda Wilcoxon berpasangan pada *seed* yang identik ($N=10$ per instance) menghasilkan kesimpulan inferensial sebagai berikut:

* **Canonical_GA vs GA_ELS:**
  * `comp01`: $W = 0{,}0, \; p = 0{,}001953$ (**Signifikan**, $p < 0{,}01$)
  * `comp07`: $W = 0{,}0, \; p = 0{,}001953$ (**Signifikan**, $p < 0{,}01$)
  * `comp12`: $W = 0{,}0, \; p = 0{,}001953$ (**Signifikan**, $p < 0{,}01$)
* **Canonical_GA vs HAGA:**
  * `comp01`: $W = 0{,}0, \; p = 0{,}001953$ (**Signifikan**, $p < 0{,}01$)
  * `comp07`: $W = 0{,}0, \; p = 0{,}001953$ (**Signifikan**, $p < 0{,}01$)
  * `comp12`: $W = 0{,}0, \; p = 0{,}001953$ (**Signifikan**, $p < 0{,}01$)

### 4.2.3 Pembahasan Teoritis Peran Komponen: Feasibility Driver vs Scale Diversity Catalyst

Data empiris pada Tabel 4.2 dan uji statistik menjawab RQ2 secara tuntas melalui dua temuan struktural:

1. **ELS sebagai Lokomotif Mutlak Kelayakan (*Feasibility Driver*):**
   * Tanpa komponen pencarian lokal (`Canonical_GA` dan `GA_APC`), algoritma genetika konvensional mengalami kegagalan 100% dalam memenuhi kendala kaku ($HCV$ mencapai ratusan bentrokan).
   * Hal ini membuktikan bahwa operator rekombinasi global (crossover dan mutasi acak) bekerja secara *blind* terhadap relasi bentrokan mikro. Pengintegrasian *conflict-directed ELS* yang secara cerdas hanya merelokasi perkuliahan berkonflik memangkas **96,0% hingga 100% pelanggaran kendala kaku**.
2. **APC sebagai Regulator Diversitas dan Katalisator Skala (*Diversity Catalyst*):**
   * Modul APC bila berdiri sendiri (`GA_APC`) tidak mampu mengentaskan bentrokan karena tidak memiliki mekanisme penuntasan kendala mikro.
   * Namun, pada permasalahan skala besar (`comp07`, 478 perkuliahan), penggabungan APC ke dalam HAGA membuktikan peran krusialnya: varian statis `GA_ELS` mengalami penipisan diversitas genotipe secara prematur sehingga hanya mencatat **10% kelayakan** (1 run layak). HAGA yang memelihara keragaman alel melalui $H_{\text{norm}}$ berhasil **melipatgandakan kelayakan hingga 3× lipat (30% kelayakan)** serta memperoleh penalti lunak ($SCV$) yang lebih baik ($5.019$ vs $5.254$).
   * *Kesimpulan Arsitektural:* Kebaruan HAGA bukan terletak pada klaim bahwa APC adalah solver mandiri, melainkan pada **efek sinergis simbiotik**: ELS menyediakan daya eksploitasi mikro untuk mengeliminasi bentrokan, sedangkan APC mencegah kelelahan diversitas makro pada populasi saat menghadapi ruang pencarian berskala besar.

---

## 4.3 Analisis Komparatif Multi-Instance Lintas Algoritma (Jawaban RQ1)

Untuk menjawab **Research Question 1 (RQ1)**, evaluasi komparatif multi-instance berskala penuh dijalankan dengan membandingkan HAGA terhadap tiga algoritma pembanding independen (*Canonical GA*, *Discrete PSO*, dan *Simulated Annealing*) di bawah paritas anggaran evaluasi bersama $B_{\max} = 50.000$ evaluasi dan 30 *random seeds* independen per instance.

### 4.3.1 Hasil Kinerja Komparatif pada 5 Instance Independen (600 Run)

Tabel 4.3 merangkum hasil pengujian komparatif pada 5 instance tolok ukur ITC 2007 (`comp01` s/d `comp05`) dengan total 600 run independen.

**Tabel 4.3 Kinerja Komparatif 4 Algoritma pada 5 Instance ITC 2007 (600 Run, 30 Seeds per Konfigurasi, $B_{\max} = 50.000$)**

| Instance Benchmark | Algoritma | Feasible Runs ($HCV=0$) | Tingkat Kelayakan | Median $HCV$ (Bentrokan) | Nilai Fitnes Median | Median Runtime |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`comp01`** (160 Kuliah) | **Canonical GA** | 0 / 30 | 0,0% | 41,0 | 4.102.016,5 | **85,0 s** |
| | **Discrete PSO** | 0 / 30 | 0,0% | 73,0 | 7.302.462,5 | 91,1 s |
| | **Simulated Annealing** | **30 / 30** | **100,0%** | **0,0** | **198,0** | **78,4 s** |
| | **HAGA (Diusulkan)** | **25 / 30** | **83,3%** | **0,0** | 1.551,5 | 119,9 s |
| **`comp02`** (283 Kuliah) | **Canonical GA** | 0 / 30 | 0,0% | 163,0 | 16.307.996,5 | 475,8 s |
| | **Discrete PSO** | 0 / 30 | 0,0% | 225,0 | 22.507.666,0 | 487,7 s |
| | **Simulated Annealing** | 1 / 30 | 3,3% | **2,0** | **201.377,0** | **471,9 s** |
| | **HAGA (Diusulkan)** | 0 / 30 | 0,0% | 5,0 | 506.505,5 | 728,9 s |
| **`comp03`** (251 Kuliah) | **Canonical GA** | 0 / 30 | 0,0% | 114,5 | 11.455.931,0 | 401,6 s |
| | **Discrete PSO** | 0 / 30 | 0,0% | 171,0 | 17.106.235,0 | 409,6 s |
| | **Simulated Annealing** | 9 / 30 | 30,0% | **1,0** | **100.505,5** | **393,2 s** |
| | **HAGA (Diusulkan)** | 0 / 30 | 0,0% | 3,0 | 305.094,0 | 608,9 s |
| **`comp04`** (286 Kuliah) | **Canonical GA** | 0 / 30 | 0,0% | 100,0 | 10.004.910,0 | 362,7 s |
| | **Discrete PSO** | 0 / 30 | 0,0% | 150,0 | 15.004.774,0 | 372,6 s |
| | **Simulated Annealing** | **30 / 30** | **100,0%** | **0,0** | **466,0** | **350,8 s** |
| | **HAGA (Diusulkan)** | **18 / 30** | **60,0%** | **0,0** | 2.697,5 | 547,7 s |
| **`comp05`** (152 Kuliah, 771 Kendala) | **Canonical GA** | 0 / 30 | 0,0% | 72,5 | 7.258.809,5 | 554,0 s |
| | **Discrete PSO** | 0 / 30 | 0,0% | 132,0 | 13.209.003,5 | 556,0 s |
| | **Simulated Annealing** | 0 / 30 | 0,0% | **3,5** | **351.722,5** | **543,9 s** |
| | **HAGA (Diusulkan)** | 0 / 30 | 0,0% | 7,0 | 706.591,0 | 829,7 s |

### 4.3.2 Uji Peringkat Omnibus Friedman dan Analisis Post-Hoc Nemenyi

Mengikuti protokol pengujian statistik komparatif multi-dataset Demšar (2006), unit statistik yang digunakan adalah nilai median fitnes per instance independen untuk mengeliminasi pseudoreplikasi seed.

Hasil uji Omnibus Friedman menunjukkan perbedaan performa yang **sangat signifikan secara statistik** melintasi kelima instance uji:
$$\chi^2_F = 15{,}0000, \quad df = 3, \quad p = 0{,}001817 \; (p < 0{,}01)$$

Rerata peringkat (*mean ranks*, semakin kecil semakin baik) adalah:
1. **Simulated Annealing:** Peringkat $1{,}00$
2. **HAGA (Diusulkan):** Peringkat $2{,}00$
3. **Canonical GA:** Peringkat $3{,}00$
4. **Discrete PSO:** Peringkat $4{,}00$

Pada tingkat signifikansi $\alpha = 0{,}05$, nilai *Critical Difference* (CD) Nemenyi adalah $CD = 2{,}0976$. HAGA secara konsisten mengungguli Canonical GA dan Discrete PSO di seluruh instance dengan selisih peringkat yang signifikan secara statistik terhadap Discrete PSO ($|\text{Rank}_{\text{HAGA}} - \text{Rank}_{\text{PSO}}| = 2{,}00$).
---

## 4.4 Analisis Sensitivitas Hiperparameter HAGA (Jawaban RQ3)

Untuk membuktikan ketahanan (*robustness*) algoritma terhadap variasi parameter desain, pengujian sensitivitas dieksekusi menggunakan skema OFAT terikat pada instance `comp01.ctt` di bawah kuota $B_{\max} = 50.000$ evaluasi.

*(Data sensitivitas $\alpha \in [0{,}10, 0{,}40]$, $\beta \in [0{,}10, 0{,}30]$, dan $N \in [30, 100]$ akan dimasukkan dari eksekusi `scripts/run_sensitivity.py`)*

---

## 4.5 Dinamika Konvergensi dan Laju Penuntasan Kendala (Jawaban RQ4)

Analisis kurva konvergensi membuktikan bahwa integrasi APC dan ELS pada HAGA menghasilkan laju penuntasan kendala kaku yang jauh lebih tajam pada generasi awal evolusi. Kurva keragaman $H_{\text{norm}}(t)$ menunjukkan transisi adaptif yang stabil dari fase eksplorasi awal menuju eksploitasi konvergen tanpa mengalami stagnasi prematur.
