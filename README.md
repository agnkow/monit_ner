# monitner
## Monitoring modeli NER (pl-spacy)

### Dryf danych

* Średnia długość tekstu:
  - średnia liczba tokenów
  - średnia liczba słów
  - średnia liczba zdań
 <p></p>

* Procentowy udział w tekście:
  - słów
  - cyfr
  - interpunkcji
  - słów rozpoczynających się wielką literą
  - tagów POS
 <p></p>

* Test Kolmogorova-Smirnova dla:
  - rozkładu liczby tokenów na zdanie
  - rozkładu liczby encji na zdanie
 <p></p>


&#8203;
### Dryf emeddingów
- cosine similarity między centroidami embeddingów
- zmiana semantyki:
	- tekstów (wszystkie tokeny)
	- kontekstów encji (tokeny nie będące encjami)
	- encji 


&#8203;
### Dryf predykcji

Dryf rozkładu encji
- KL Divergence (Kullback–Leibler) [0, +∞)
- Jensen–Shannon Distance [0, 1]

* zmiana proporcji:
  - persName/placeName/orgName/geogName



&#8203;
### Rozpoznanie

1. Zmiana tematyki tekstów
   - wzrost Jensen–Shannon Distance dla rozkładu encji
   - wzrost KL Divergence dla rozkładu encji
   - wysoki poziom dryfu embeddingów dla wszystkich tokenów
 <p></p>

2. Pojawienia się nowych typów encji
   - spadek średniej liczby encji na liczbę tokenów 
   - zmiana średniej długości encji 
   (także w poszczególnych kategoriach persName/placeName/orgName/geogName)
   - wysoki poziom dryfu embeddingów dla encji 
 <p></p>

3. Zmiana stylu tekstów
   - zmiana średniej liczby encji na liczbę tokenów  
   - zmiana średniej liczby tokenów na zdanie (np. styl formalny/nieformalny)
   - zmiana % udziału interpunkcji w tekstach (jw.)
   - zmiana % udziału liczb w tekstach (np. styl mniej lub bardziej biznesowy, finansowy)
   - zmiana rozkładu tagów POS (zmiana rejestru językowego)
   - wysoki poziom dryfu embeddingów dla wszystkich tokenów oraz tokenów nie będących encjami




&#8203;

Źródła:
* Explainable Data Drift for NLP, NLP Summit 2023, [link](https://www.youtube.com/watch?v=HnHkW_M3e6U)
* Domain Divergences: A Survey and Empirical Analysis, 
Authors: Abhinav Ramesh Kashyap, Devamanyu Hazarika, 
Min-Yen Kan, Roger Zimmermann, 23.10.2020 [link](https://arxiv.org/abs/2010.12198)
