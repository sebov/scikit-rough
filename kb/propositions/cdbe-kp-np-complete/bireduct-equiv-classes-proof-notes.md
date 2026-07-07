# Dowód: |X/B| ≥ |B| + 1 dla decision bireduct

## Status
- **Kroki 1-4**: Zapisane w TeX (`prop:bireduct_equiv_classes_geq_bplus1`, lines 480-517)
- **Ten plik**: Do skasowania po przeniesieniu wyniku do KB

## Ustalony dowód (4 kroki)

### Krok 1 (gotowy w TeX)
Niech m = |X/B| będzie liczbą klas abstrakcji. Ponieważ (X, B) jest bireductem, B jest nieredukowalne: dla każdego b ∈ B, usunięcie b łamie zależność funkcyjną. Zatem dla każdego b ∈ B istnieją u, v ∈ X różniące się tylko na b. Ponieważ się różnią, należą do różnych klas abstrakcji. Niech C_b i D_b będą ich klasami. Te dwie klasy różnią się tylko na atrybucie b (są zgodne na B \ {b}).

### Krok 2 (gotowy w TeX)
Zbuduj graf H: wierzchołki = klasy abstrakcji, krawędzie = {(C_b, D_b) : b ∈ B}. H ma m wierzchołków i |B| krawędzi, każda oznaczona innym b.

### Krok 3 (do zapisania)
H jest lasem. Załóżmy, że istnieje cykl. Każda krawędź oznaczona b zmienia tylko wartość na pozycji b w wektorze reprezentatywnym klasy; wszystkie inne pozycje są zachowane. Idąc wzdłuż cyklu, aby wrócić do klasy początkowej, wartość każdej pozycji musi wrócić do wartości początkowej. Ale każde b oznacza dokładnie jedną krawędź w H, więc każdy atrybut b występujący w cyklu miałby swoją wartość zmienioną dokładnie raz — docierając do innej wartości niż początkowa. Sprzeczność. Zatem cykl nie istnieje.

### Krok 4 (do zapisania)
W lesie |E| ≤ |V| - 1. Przy |E| = |B| i |V| = m otrzymujemy |B| ≤ m - 1, stąd m ≥ |B| + 1.

## Uwagi
- Dowód działa dla dowolnych atrybutów (nie tylko binarnych)
- Nie wymaga `bireduct-attrs-subset-form-bireduct`
- Argument grafowy jest naturalnym językiem tego faktu
- **Wynik jest OGÓLNY** - dotyczy dowolnych bireduktów, nie tylko specyficznych dla CDBEkP
- Powinien trafić do głównego katalogu `kb/propositions/`, nie do podkatalogu `cdbe-kp-np-complete/`
- Analogicznie `prop-bireduct-desc-len-geq-bplus1-squared` (aktualnie w staging/) jest wynikiem ogólnym

## Następne kroki
1. ~~Dokończyć zapis kroków 2-4 w TeX~~ (zrobione, wszystkie 4 kroki w main.tex)
2. ~~Przenieść `prop-bireduct-equiv-classes-geq-bplus1` z staging/ do głównego katalogu propositions/~~ (zrobione)
3. Przenieść `prop-bireduct-desc-len-geq-bplus1-squared` z staging/ do głównego katalogu propositions/
4. Skasować ten plik notes
5. Wrócić do `prop:bireduct_replacement_correctness_and_simpler` - dokończyć dowód nierówności desc-len
