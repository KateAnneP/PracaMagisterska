### --- Users - tabela i przygotowane dane do używania ---
from datetime import datetime
import pandas as pd
from pandas import CategoricalDtype

#Jest tutaj warunek, że jak ktoś nie ma partnera, to nie jest brany pod uwagę

current_year = datetime.now().year  # Bieżący rok

users = pd.read_csv('excluded/dane/ausers.csv', delimiter=';')
prepared = users.copy()

# ------------------------------------------------------------------------------
do_usuniecia = []
prepared['partner_education'] = None
prepared['partner_age'] = None

for i in range(len(prepared)):
    # if prepared['gender'].iloc[i] == "M":
    id = prepared['email']
    partner = prepared['partner_email'].iloc[i]
    znaleziono_partnera = False
    for j in range(len(prepared)):
        if prepared['email'].iloc[j] == partner:
            znalezniono_partnera = True
            partner_wiek = current_year - prepared['birth_year'].iloc[j]
            partner_edukacja = prepared['education'].iloc[j]
    if znaleziono_partnera == False:
        prepared.loc[i, 'partner_education'] = partner_edukacja
        prepared.loc[i, 'partner_age'] = partner_wiek
    else:
        do_usuniecia.append(i)

prepared = prepared.drop(prepared.index[do_usuniecia]).reset_index(drop=True)  # Jak nie znaleziono partnera, to nie ma sensu klasyfikować tej osoby w tym zestawieniu
# ------------------------------------------------------------------------------

# Eliminowanie NaN
for i in range(len(prepared)):
    if pd.isna(prepared['doughter_birth_years'].iloc[i]):
        prepared.loc[i, 'doughter_birth_years'] = "0"
    if pd.isna(prepared['son_birth_years'].iloc[i]):
        prepared.loc[i, 'son_birth_years'] = "0"

# Ilość synów i córek
prepared['no_of_daughters'] = prepared['doughter_birth_years'].apply(lambda x: 0 if x == '0' else len(x.split(',')))
prepared['no_of_sons'] = prepared['son_birth_years'].apply(lambda x: 0 if x == '0' else len(x.split(',')))

# Wiek najmłodszego dziecka
prepared['child_birth_years'] = prepared['doughter_birth_years'].fillna('') + ',' + prepared['son_birth_years'].fillna('')
prepared['child_birth_years'] = prepared['child_birth_years'].str.strip(',')
prepared['age_oldest_child'] = current_year - prepared['child_birth_years'].apply(lambda x: max(map(int, x.split(','))))
prepared['age_oldest_child'] = prepared['age_oldest_child'].apply(lambda x: 0 if x == current_year else x)

# Średnia wieku dzieci
def calculate_average_years(date_string):
    years = date_string.split(',')  # Daty rozdzielone przecinkami
    age_differences = [current_year - int(year.strip()) for year in years if year.strip().isdigit() and year != '0']
    return round(sum(age_differences) / len(age_differences)) if age_differences else 0

prepared['average_child_age'] = prepared['child_birth_years'].apply(calculate_average_years).apply(lambda x: 0 if x == current_year else x)

# Edukacja
education_order = CategoricalDtype(categories=['średnie', 'wyższe'], ordered=True)  # 0 - średnie, 1 - wyższe
prepared['education'] = prepared['education'].astype(education_order)
prepared['education'] = prepared['education'].cat.codes
prepared['partner_education'] = prepared['partner_education'].astype(education_order)
prepared['partner_education'] = prepared['partner_education'].cat.codes

# Mapowanie płci na liczby
gender_mapping = {'K': 0, 'M': 1}
prepared['gender'] = prepared['gender'].map(gender_mapping)

# Kolumna rok urodzenia - zamiana na wiek
prepared['age'] = current_year - prepared['birth_year']  # Tworzenie nowej kolumny z wiekiem
prepared['years_of_mariage'] = current_year - prepared['mariage_year']  # Kolumna ze stażem małżeństwa
prepared['years_of_relationship'] = current_year - prepared['relation_start_year']  # Kolumna ze stażem związku

# Usuwanie niepotrzebnych do grupowania kolumn
prepared = prepared.drop(columns=['birth_year', 'doughter_birth_years', 'son_birth_years', 'child_birth_years', 'mariage_year', 'relation_start_year'])
#'email', 'partner_email',
#print(list(prepared))