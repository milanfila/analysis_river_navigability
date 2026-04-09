# Minimalistický Streamlit dashboard pro Oslavu

## Lokální spuštění
pip install -r requirements_dashboard.txt
streamlit run app_oslava_dashboard.py

## Co ukazuje
- stav JEĎ / SLEDUJ / NEJEĎ
- pravděpodobnost sjízdnosti za cca 2 hodiny
- H Mostiště
- dH Mostiště za 1 hodinu
- H Nesměř
- graf posledních 48 hodin

## Důležité
V sidebaru jsou zatím ukázkové koeficienty logistického modelu.
Až budeš mít svoje finální koeficienty, přepiš je tam nebo je natvrdo vlož do DEFAULT_MODEL.
