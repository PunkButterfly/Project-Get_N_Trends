from .config import *


# Функция для парсинга
async def get_channel_info(channel_name: str, api_id: int, api_hash: int,
                           parsed_last_time: datetime = None, limit: int = None):
    async with TelegramClient('data/session', api_id, api_hash) as client:
        channel = await client.get_entity(channel_name)
        # date = pd.to_datetime(parsed_last_time, format = date_format)
        reverse = True
        if not parsed_last_time:
            reverse = False
        messages = await client.get_messages(channel, offset_date=parsed_last_time, limit=limit, reverse=reverse, wait_time=2)
        return (messages)
