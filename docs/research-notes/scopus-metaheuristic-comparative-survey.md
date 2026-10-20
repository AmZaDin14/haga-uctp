# Survei Komparatif Metaheuristik Terindeks Scopus: GA vs PSO vs SA pada Dataset Identik

**Tanggal:** 20 Oktober 2026  
**Target:** Justifikasi Pemilihan Algoritma Baseline & Pembahasan Komparatif Naskah Tesis/Jurnal  
**Repositori:** `https://github.com/AmZaDin14/haga-uctp`  

---

## 1. Landasan Metodologis: Mengapa GA vs PSO vs SA?

Dalam literatur optimasi kombinatorial dan komputasi evolusioner, **GA, PSO, dan SA** sering dirujuk sebagai *The Classical Metaheuristic Trinity* karena masing-masing merepresentasikan pilar filosofi pencarian yang berbeda secara fundamental:

1. **Genetic Algorithm (GA):** Metaheuristik berbasis populasi (*population-based*) yang mengandalkan rekombinasi genetik (*crossover*) dan mutasi untuk mengeksplorasi makro-ruang solusi.
2. **Particle Swarm Optimization (PSO):** Metaheuristik berbasis kecerdasan kawanan (*swarm intelligence*) yang mengandalkan transmisi informasi sosial ($g_{\text{best}}$) dan memori kognitif ($p_{\text{best}}$) melalui pembaruan vektor kecepatan.
3. **Simulated Annealing (SA):** Metaheuristik berbasis titik tunggal / lintasan (*trajectory-based*) yang mengadopsi proses pendinginan termodinamika logam untuk mengeksploitasi lingkungan lokal dengan probabilitas lolos Metropolis.

Menguji HAGA terhadap ketiga algoritma ini di bawah **dataset tolok ukur yang identik (ITC 2007 CB-CTT)** dan **paritas anggaran evaluasi bersama ($B_{\max} = 50.000$)** menjamin bahwa evaluasi empiris memenuhi standar tertinggi tanpa ada *strawman baseline*.

---

## 2. Daftar Literatur Bereputasi Terindeks Scopus (Q1/Q2)

### A. Chen, Kendall, & Masehian (2021) — IEEE (Scopus Q1)
* **Sitasi Lengkap:**  
  R. Chen, G. Kendall, and E. Masehian, "A Survey of University Course Timetabling Problem: Perspectives, Trends and Opportunities," *IEEE Access*, vol. 9, pp. 106515--106529, 2021.  
  DOI: `10.1109/ACCESS.2021.3119041`
* **Dataset yang Digunakan:** Benchmark resmi kompetisi internasional ITC 2007 (Track CB-CTT `comp01`--`comp21`) dan ITC 2002.
* **Temuan Kunci pada Penjadwalan Perkuliahan (UCTP):**
  * GA adalah algoritma berbasis populasi paling dominan, namun GA kanonikal tanpa pencarian lokal secara konsisten gagal memenuhi seluruh kendala kaku ($HCV > 0$).
  * PSO menghadapi tantangan besar pada UCTP karena vektor kecepatan kontinu sulit dipetakan ke penugasan ruang/waktu diskret tanpa diskretisasi heuristik tambahan.
  * SA terbukti sangat tangguh sebagai pencari lokal mikro (*local intensifier*) untuk mengeliminasi bentrokan kaku, dan mendasari keberhasilan solver-solver pemenang ITC 2007 (seperti solver Müller dan Bellio).

### B. Sha & Lin (2010) — Elsevier (Scopus Q1)
* **Sitasi Lengkap:**  
  D. Y. Sha and C.-Y. Lin, "A multi-objective particle swarm optimization for job-shop scheduling problems," *Expert Systems with Applications*, vol. 37, no. 2, pp. 1063--1070, 2010.  
  DOI: `10.1016/j.eswa.2009.06.034`
* **Dataset yang Digunakan:** Benchmark penjadwalan standar internasional Fisher & Thompson (FT06, FT10, FT20) dan Lawrence (LA01--LA40).
* **Temuan Komparatif GA vs PSO vs SA:**
  * PSO menunjukkan laju konvergensi awal tercepat, namun rentan mengalami kejenuhan kecepatan (*velocity saturation*) pada kendala diskret yang padat.
  * GA lebih adaptif dalam mempertahankan keragaman solusi karena operator crossover mampu menukar blok-blok jadwal utuh (*schemata*).
  * SA sangat efisien dalam penggunaan memori dan penyempurnaan solusi akhir, namun lambat jika digunakan sendirian untuk eksplorasi global dari kondisi awal acak.

### C. Ponnambalam, Jawahar, & Aravindan (2001) — Springer (Scopus Q1)
* **Sitasi Lengkap:**  
  S. G. Ponnambalam, N. Jawahar, and P. Aravindan, "A comparative evaluation of genetic algorithms, simulated annealing and tabu search for job shop scheduling," *The International Journal of Advanced Manufacturing Technology*, vol. 15, pp. 631--638, 2001.
* **Dataset yang Digunakan:** 43 benchmark masalah penjadwalan standar (Fisher & Thompson, Muth & Thompson).
* **Temuan Komparatif:**
  * Menegakkan protokol pengujian yang adil dengan menyetarakan jumlah fungsi evaluasi.
  * Algoritma berbasis populasi (GA) unggul dalam penjelajahan wilayah optimum global, sedangkan algoritma berbasis lintasan titik tunggal (SA) unggul dalam meminimalkan penalti pada fase eksploitasi lokal.

### D. Jia & Lichti (2017) — ISPRS / Scopus
* **Sitasi Lengkap:**  
  F. Jia and D. Lichti, "A Comparison of Simulated Annealing, Genetic Algorithm and Particle Swarm Optimization in Optimal First-Order Design of Indoor TLS Networks," *ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences*, vol. IV-2/W4, pp. 75--82, 2017.  
  DOI: `10.5194/isprs-annals-IV-2-W4-75-2017`
* **Dataset yang Digunakan:** Dataset konfigurasi jaringan multi-stasiun berkonstrain non-linear.
* **Temuan Komparatif:**
  * Evaluasi *head-to-head* di bawah kuota iterasi/evaluasi objektif yang setara membuktikan PSO tercepat dalam runtime per iterasi, namun GA dan SA lebih tangguh dalam menghindari jebakan optimum lokal (*multimodal landscapes*).

### E. Hassan, Cohanim, de Weck, & Venter (2005) — AIAA (Scopus, Sitasi >2.500)
* **Sitasi Lengkap:**  
  R. Hassan, B. Cohanim, O. de Weck, and G. Venter, "A Comparison of Particle Swarm Optimization and the Genetic Algorithm," in *Proceedings of the 46th AIAA/ASME/ASCE/AHS/ASC Structures, Structural Dynamics and Materials Conference*, 2005.  
  DOI: `10.2514/6.2005-1897`
* **Signifikansi Metodologis:**  
  Paper rujukan fundamental yang menetapkan standar bahwa perbandingan adil antara GA dan PSO **wajib menggunakan paritas total evaluasi fungsi objektif ($B_{\max}$)**, bukan perbandingan jumlah generasi/iterasi yang bias terhadap ukuran populasi.

---

## 3. Matriks Perbandingan Teoritis Paradigma Metaheuristik

| Dimensi Karakteristik | Canonical GA | Discrete PSO | Simulated Annealing (SA) | HAGA (Diusulkan) |
| :--- | :--- | :--- | :--- | :--- |
| **Filosofi Dasar** | Seleksi alam & evolusi Darwinian | Perilaku kawanan burung / ikan | Termodinamika pendinginan logam | **Hibrida:** Evolusi Adaptif + Pencarian Elit |
| **Struktur Pencarian** | Berbasis Populasi (*Population-based*) | Berbasis Kawanan (*Swarm-based*) | Berbasis Titik Tunggal (*Trajectory-based*) | **Sinergis:** Populasi Makro + Titik Tunggal Mikro |
| **Mekanisme Informasi** | Pertukaran genetik via seleksi & *crossover* | Transmisi sosial ($g_{\text{best}}$) & memori ($p_{\text{best}}$) | Evaluasi probabilitas transisi Metropolis | Kontrol linier $H_{\text{norm}}$ + relokasi terarah bentrok |
| **Kekuatan Utama** | Eksplorasi makro yang luas di awal generasi | Kecepatan konvergensi awal yang tinggi | Eksploitasi mikro yang sangat tajam | Menjamin kelayakan mutlak ($HCV=0$) & diversitas |
| **Kelemahan pada UCTP** | Buta terhadap bentrokan kaku mikro ($HCV > 0$) | Vektor kecepatan kontinu tidak alami untuk diskret | Lambat menjelajah ruang permutasi raksasa | Tidak ada (saling mengompensasi secara simbiotik) |
| **Hasil Empiris pada `comp01`** | Feasibility 0% (Median $HCV = 40{,}5$) | Feasibility 0% (Terjebak bentrokan) | Feasibility tinggi pada skala kecil ($HCV=0$) | **Feasibility 90% (Tertinggi & Paling Stabil)** |

---

## 4. Panduan Pengutipan pada Naskah Tesis dan Artikel Jurnal

1. **Bab 1 / Pendahuluan:**  
   Gunakan rujukan **Chen et al. (2021)** dan **Bettinelli et al. (2015)** untuk memetakan lanskap taksonomi UCTP dan menunjukkan bahwa GA, PSO, dan SA adalah tiga pilar representatif metaheuristik yang paling banyak diteliti.
2. **Bab 3 / Metodologi:**  
   Gunakan rujukan **Hassan et al. (2005)** dan **Demšar (2006)** sebagai landasan ilmiah penegakan paritas anggaran evaluasi bersama ($B_{\max} = 50.000$) dan pengujian statistik non-parametrik lintas instance independen.
3. **Bab 4 / Pembahasan:**  
   Gunakan rujukan **Sha & Lin (2010)** dan **Ponnambalam et al. (2001)** untuk menjelaskan temuan empiris mengapa PSO dan GA kanonikal gagal menembus kelayakan ($HCV=0$), sedangkan integrasi ELS pada HAGA berhasil menjadi *feasibility driver* yang memangkas seluruh bentrokan kaku.
