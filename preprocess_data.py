import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_and_add_features(file_path: str) -> pd.DataFrame:
    # Загрузка датасета
    df = pd.read_csv(file_path, sep=';')
    
    # Базовая информация
    print("Shape before preprocessing:", df.shape)
    
    # Преобразуем некоторые числовые признаки
    # Например, Age_years нормализуем
    scaler = StandardScaler()
    df['Age_scaled'] = scaler.fit_transform(df[['Age_years']])
    
    # Добавим новые фичи
    # 1. Отношение суммы кредита к доходу (грубо через Age_years, так как дохода нет)
    df['Credit_per_Age'] = df['Credit_Amount'] / (df['Age_years'] + 1)  # +1 чтобы не делить на 0
    
    # 2. Долговая нагрузка: сумма кредита на длительность кредита
    df['Credit_per_Duration'] = df['Credit_Amount'] / (df['Duration_of_Credit_monthly'] + 1)
    
    # 3. Индикатор: молодой заемщик с большой суммой
    df['Young_and_Big_Credit'] = ((df['Age_years'] < 25) & (df['Credit_Amount'] > 2000)).astype(int)
    
    # 4. Индикатор наличия поручителя
    df['Has_Guarantor'] = (df['Guarantors'] != 1).astype(int)
    
    # 5. Суммарный риск: плохая история + большая сумма + длительный срок
    df['Risk_Score'] = (
        (df['Payment_Status_of_Previous_Credit'] > 2).astype(int) +
        (df['Credit_Amount'] > 2500).astype(int) +
        (df['Duration_of_Credit_monthly'] > 24).astype(int)
    )
    
    print("Shape after preprocessing:", df.shape)
    
    return df
