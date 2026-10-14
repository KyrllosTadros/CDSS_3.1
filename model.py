import random # willekeurige operaties
import numpy as np # wiskundige operaties
import pandas as pd # lezen van csv-bestanden
import gdown # downloaden van Google Drive-bestanden
import matplotlib.pyplot as plt #plotten
from sklearn.neural_network import MLPClassifier #Neurale Netwerken
import os #bestandsbeheer
from sklearn.impute import SimpleImputer # voor ontbrekende waarden

def maak_dummies(df, kolommen, drop_first=True):
    # 1. Zorg ervoor dat alle opgegeven kolommen eerst als string/tekst worden gezien
    for kol in kolommen:
        if kol in df.columns:
            df[kol] = df[kol].astype(str)

    # 2. Maak de dummies aan
    # dtype=int zorgt ervoor dat de kolommen direct 0 en 1 worden (in plaats van True/False)
    df_met_dummies = pd.get_dummies(
        df, columns=kolommen, prefix=kolommen, drop_first=drop_first, dtype=int)

    return df_met_dummies

# functie voor het normaliseren van de data
def normalize(x):
    if min(x) < max(x):
        z = (x - min(x)) / (max(x) - min(x))
    elif min(x) == max(x):
        z = 1
    return z

# Functie voor het berekenen en eventueel printen van de confusion matrix
def cf_matrix(p, o, and_print=True):
    TP = sum((p == o) & (o == 1))  # True Positives
    TN = sum((p == o) & (o == 0))  # True Negatives
    FP = sum((p != o) & (o == 0))  # False Positives
    FN = sum((p != o) & (o == 1))  # False Negatives

    ER = (FP + FN) / (TP + FP + TN + FN)  # Error Rate
    AC = 1 - ER  # Accuracy

    if and_print:
        print('------------------------------')
        print('|            |Classified as: |')
        print('------------------------------')
        print('|True class: | Y*=p  | Y*=n  |')
        print('------------------------------')
        print(f'|       Y=p  |   {TP}   |   {FN}   |')
        print('------------------------------')
        print(f'|       Y=n  |   {FP}   |   {TN}   |')
        print('------------------------------')
        print(f'Error rate: {ER}')
        print(f'Accuracy: {AC}')

    return AC

def main():
    # 1. Lees het CSV-bestand in
    df = pd.read_csv(("NKR_IKNL_breast_syntheticdata.csv"), sep=";")
    print(df.describe())
    print(df.shape)

    # 2. rarget kolom aanmaken (pm, met fallback naar cm) vóór data cleaning
    df['uitkomst'] = df['pm'].combine_first(df['cm'])
    # Verwijder rijen waar de target onbekend is
    df = df.dropna(subset=['uitkomst'])
    df['uitkomst'] = df['uitkomst'].astype(int)

    # verwijder topo want overal gelijk
    df = df.drop(columns=['topo'])
    df = df.drop(columns=['key_nkr', 'key_eid'])

    #3. missende waarden vervangen door NaN
    ## Verander 9 bij diffgrad naar nan
    df['diffgrad'] = df['diffgrad'].replace(9, np.nan)

    ## Verander M bij stadium, pstadium en cstadium naar nan
    df['stadium'] = df['stadium'].replace('M', np.nan)
    df['pstadium'] = df['pstadium'].replace('M', np.nan)
    df['cstadium'] = df['cstadium'].replace('M', np.nan)

    ## Verander 98 en 99 bij ond_lymf  naar nan
    df['ond_lymf'] = df['ond_lymf'].replace([98, 99], np.nan)

    ## Verander 99 bij pos_lymf naar nan
    df['pos_lymf'] = df['pos_lymf'].replace(99, np.nan)

    ## Verander 9 bij er_stat naar nan
    df['er_stat'] = df['er_stat'].replace(9, np.nan)

    ## Verander 9 bij pr_stat naar nan
    df['pr_stat'] = df['pr_stat'].replace(9, np.nan)

    ## Verander 4 en 7 bij her2_stat naar nan
    df['her2_stat'] = df['her2_stat'].replace([4, 7], np.nan)

    ## Verander 9 bij multifoc naar nan
    df['multifoc'] = df['multifoc'].replace(9, np.nan)

    ## Verander 8.0 bij swk naar nan
    df['swk'] = df['swk'].replace(8.0, np.nan)

    ## Verander 9.0 bij swk_uitslag naar nan
    df['swk_uitslag'] = df['swk_uitslag'].replace(9.0, np.nan)

    #3. Percentage missing values per kolom berekenen
    ## Bekijk welke variabelen ergens een NaN waarde hebben
    missing_vars = df.columns[df.isna().any()].tolist()
    print(f"Variabelen met missings: {missing_vars}")

    ## Bekijk percentages missings voor elke variabele
    missing_percentages = (df.isna().mean() * 100)
    print("\nPercentages missings per variabele:")
    print(missing_percentages)

    ## Variabelen met meer dan 25% missings
    missing_25 = missing_percentages[missing_percentages > 25].index

    ## Excludeer kolommen met meer dan 25% missings
    df.drop(columns=missing_25, inplace=True)

    # 4. Imputeer de rijen die onbekend zijn:
    df = df.copy()
    imp = SimpleImputer(strategy="most_frequent")
    df[:] = imp.fit_transform(df)

    ## Bekijk percentages missings voor elke variabele
    missing_percentages = (df.isna().mean() * 100)
    print("\nPercentages missings per variabele:")
    print(missing_percentages)

    # 5. Categorische variabelen one-hot-encoden
    ## Lijst met alle kolommen die we willen one-hot-encoden
    categorische_kolommen = [
        'diag_basis',
        'ct',
        'cn',
        'pt',
        'pn',
        'stadium',
        'cstadium',
        'pstadium',
        'diffgrad',
        'uitgebr_chir_code',
        'gedrag',
        'gesl',
        'her2_stat',
        'dcis_comp',
        'okd',
        'org_chir',
        'dir_reconstr',
        'meta_rt',
        'meta_chir',
        'multifoc',
        'later',
        'morf',
        'er_stat',
        'pr_stat',
        'swk',
        'chemo',
        'target',
        'horm',
        'rt',
        'topo_sublok',
        'tumsoort',
        'vit_stat',
    ]

    df = maak_dummies(df, kolommen=categorische_kolommen, drop_first=True)
    print("Aantal kolommen na one-hot encoding:", df.shape[1])
    print(df.head())

    # 6. Normaliseer de numerieke kolommen
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].apply(normalize)
    print("\nData na normalisatie:")
    print(df.head())

    print("HIER", df.dtypes)

    # 7. Train-test split (2/3 voor training, 1/3 voor testen)
    np.random.seed(42)
    train_size = int(2/3 * len(df))
    shuffled_indices = np.random.permutation(len(df))
    train_indices = shuffled_indices[:train_size]
    test_indices = shuffled_indices[train_size:]

    ## Selecteren van de train en test sets
    df_train = df.iloc[train_indices]
    df_test = df.iloc[test_indices]

    ## Bereken de a priori kans op overlijden in deze dataset.
    apriori = df['uitkomst'].mean()
    print(f"A-priori kans op metastasen op afstand: {apriori:.3f}")

    # 8. Train model
    nn = MLPClassifier(hidden_layer_sizes=300, max_iter=1000, random_state=42)
    nn.fit(df_train.drop(columns="uitkomst"), df_train["uitkomst"]) # data leakage door pm en cm?
    p_nn = nn.predict(df_test.drop('uitkomst', axis=1))

    # 9. Evalueer model op de test set
    p_nn_class = (p_nn > 0.5).astype(int)
    ## Kijk naar de confusion matrix
    print(cf_matrix(p_nn_class, df_test["uitkomst"]))


if __name__ == "__main__":
    main()
