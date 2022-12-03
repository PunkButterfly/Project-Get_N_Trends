from sklearn.cluster import KMeans
import pandas as pd


class KMeansClustering:
    def __init__(self, text_pool, embeddings, clusters_num=None):
        """
            Класс для кластеризации эмбеддингов.
            На вход принмает информацию о новостях, эмбеддинги и количество кластеров.
            Если оно не задано, то задаётся эмпирически.
        """
        self.embeddings = embeddings
        self.clusters_num = clusters_num
        self.text_pool = text_pool

        if clusters_num is None:
            self.clusters_num = round((float(len(embeddings))) ** (1 / 2.2))  # эвристика

    def clustering(self):
        """
            Функция возвращает выполняющая поиск центроид кластеров.
            Выход: 
                data - датафрейм с информацией о новостях, в который добавлены
            метки кластеров и эмбеддинги,
                centoids - центроиды кластеров
        """
        print("Clustering data...")
        kmeans = KMeans(n_clusters=self.clusters_num, random_state=42).fit(self.embeddings.to_list())
        kmeans_labels = kmeans.labels_

        data = pd.DataFrame()
        data['title'] = self.text_pool['title']
        data['text'] = self.text_pool['content']
        data['date'] = self.text_pool['date']
        data['channel_name'] = self.text_pool['channel_name']
        data['label'] = kmeans_labels
        data['embedding'] = self.embeddings.to_list()
        data['title_text'] = self.text_pool['title'] + '\n' + self.text_pool['content']
        centroids = kmeans.cluster_centers_

        print("Clustering done!")

        return data, centroids
