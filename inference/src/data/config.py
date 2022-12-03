from __future__ import annotations
import nest_asyncio
import asyncio
from telethon import TelegramClient, functions, types, helpers
from asyncio import run
import telethon
import pandas as pd
from datetime import datetime
import re
import os
import telethon.sync

API_ID = #YOUR API_ID
API_HASH = #YOUR API_HASH
phone_number = #YOUR PHONE_NUBER
token = #YOUR TOKEN
channel_list = ['startupoftheday','mytar_rf', 'businesstodayy',
                'exploitex', 'ostorozhno_novosti', 'techno_news_tg',
                'd_code', 'FatCat18', 'bolecon', 'AK47pfl']
dir_to_save = "./data/processed"
date_format = '%Y-%m-%d'
limit = 1000
parse_from_date = '2022-01-01'
