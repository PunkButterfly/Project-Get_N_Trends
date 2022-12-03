# Структура проекта

├── configs  
│   ├── prediction_config.yaml   <- Сonfiguration file for prediction  
├── data  
│   ├── predictions   <- Directory with predictions in json format  
│   ├── processed   <- Processed updated news articles  
├── inference  
│   ├── src  
│   │   ├── data  
│   │   │   ├── config.py   <- Сonfiguration file for parsing  
│   │   │   ├── main.py   <- Parsing, processing and writing data  
│   │   │   ├── parsing_tg_funcs.py   <- Telegram channel parsing function  
│   │   │   ├── text_preprocess_funcs.py   <- Preprocessing news articles  
├── ml_model  
│   ├── src  
│   │   ├── data  
│   │   │   ├── read_data.py    <- Reading data for a certain period of time  
│   │   │   ├── write_data.py    <- Formatting prediciton to json and saving  
│   │   ├── models  
│   │   │   ├──  clustering.py    <- Clustering model  
│   │   │   ├──  embeddings.py    <- Embedder  
│   │   │   ├──  features.py    <- Getting main entities for prediction  
│   │   │   ├──  keywords_extractor.py    <- Keywords extractor and utils for it  
│   │   ├── predict_pipeline.py   <- Pipeline for prediction  
│   │   ├── process.py   <- Getting response of digest, trends, insights  
├── .dockerignore  
├── .gitignore  
├── Dockerfile  
├── find_response.py    <- Search for a suitable predictor from existing ones  
├── main.py    <- Main API app  
├── refresh_data.py    <- Periodic parsing function  
├── requirements.txt  
├── utils.py    <- Read config, date processing  
