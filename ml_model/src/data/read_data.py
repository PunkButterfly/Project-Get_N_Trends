import pandas as pd
from utils import read_config


def read_data_by_period(path_to_data: str = "./data/processed/parsed_data.csv", start_date: str = None,
                        end_date: str = None) -> pd.DataFrame:
    """
        Выдeление новостей из заданного временного промежутка.
        Вход:
            df - DataFrame с колонками 'content' и 'date'
            start_date_string - строка задающая начало временного периода,
        в формате yyyy-mm-dd
            end_date_string - строка задающая конец временного периода.
        Выход:
            отфильтрованный по времени DataFrame
    """
    params = read_config()
    df = pd.read_csv(path_to_data)

    df.loc[:, 'date'] = pd.to_datetime(df['date'], format=params["date_format"])
    mask = (df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))
    return df.loc[mask, :]
