import json

from ml_model.src.models.clustering import KMeansClustering
from ml_model.src.models.embeddings import RuBertEmbedder
from ml_model.src.models.features import get_digest, get_trends, get_insights

from ml_model.src.data.read_data import read_data_by_period


def get_response(start_date: str = None, end_date: str = None) -> json:
    """
        Получение инсайтов, трендов и дайджеста.
        Вход:
            df - DataFrame с колонками 'content' и 'date'
            start_date - строка задающая начало временного периода,
        в формате yyyy-mm-dd
            end_date - строка задающая конец временного периода.
        Выход:
            кортеж из неотформатированных результатов (инсайты, тренды, дайджест)
    """
    # выделение новостей из конкретного временного периода
    text_pool = read_data_by_period(start_date=start_date, end_date=end_date)

    # получение эмбеддингов из содержимого новостей
    embeddings_pool = RuBertEmbedder().encode_data(text_pool, data_column='content')

    clustering_data, centroids = KMeansClustering(text_pool, embeddings_pool).clustering()  # кластеризация
    centroids_map = {i: centroids[i] for i in range(len(centroids))}

    digest = get_digest(clustering_data, centroids_map, top_clusters=3)  # дайджест
    trends = get_trends(clustering_data, centroids_map, top_for_cluster=5, max_news_len=80)  # тренды
    insights = get_insights(clustering_data, centroids_map, top_for_cluster=30, max_news_len=100)  # инсайты

    return digest, trends, insights
