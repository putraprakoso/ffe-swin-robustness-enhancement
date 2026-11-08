# Panduan singkat untuk peneliti (Bahasa Indonesia)

Repositori ini adalah **paket pendukung artikel JRI**, bukan seluruh skripsi. Notebook dan hasil eksperimen sudah disaring sesuai fokus artikel: dampak kualitas citra terhadap classifier Swin-Tiny dan apakah koreksi Gamma/CLAHE membantu foto gelap.

## Apa yang sudah ada?

- `notebooks/`: kode penelitian; **hasil output sel sudah dibersihkan** agar tidak membawa data akun dan agar ukurannya kecil.
- `results/08_robustness/`: tabel dan grafik percobaan gelap, terang, blur.
- `results/09_enhancement/`: tabel dan grafik koreksi Gamma dan CLAHE.
- `notebooks/ffe_core.py`: fungsi bersama untuk dataset, split, model, dan metrik.

## Bagaimana memakainya?

1. Untuk **membaca hasil**, buka `README.md`, `docs/RESULTS.md`, dan gambar di `results/`. Tidak perlu menjalankan Colab lagi.
2. Untuk **mengulangi eksperimen dengan Drive yang sekarang**, notebook 08 dan 09 dapat dijalankan dari workspace Drive seperti biasa; hasil lama dan checkpoint tetap berada di Drive dan sengaja tidak dipindahkan ke GitHub.
3. Untuk **reproduksi sepenuhnya**, siapkan dataset ZIP resmi di `01_data/raw`, salin `ffe_core.py` ke `02_notebooks/`, lalu jalankan 00, 02, 03, 04_deep_fcn_only, 08, 09 secara berurutan. Membuat ulang backbone membutuhkan GPU.
4. Pastikan `FFE_PROJECT` atau path PROJECT pada notebook menunjuk ke folder penelitian yang tepat.
5. Periksa `results/` dan laporan kompatibilitas; jika hasil berbeda, jangan langsung menganggap reproduksi identik.

## Hal yang tidak dimasukkan

Dataset mentah, PDF artikel berhak cipta, skripsi pribadi, token GitHub/Google, checkpoint >100 MB, notebook fusion/XAI yang tidak dibahas dalam artikel.

## Catatan untuk submission JRI

Gunakan repositori **private** sampai kebijakan *double-blind review* dan arahan pembimbing jelas. Jangan menampilkan nama/afiliasi penulis pada suplemen yang harus anonim. Repositori ini mendokumentasikan eksperimen, bukan bukti model aman menilai kelayakan konsumsi ikan.
