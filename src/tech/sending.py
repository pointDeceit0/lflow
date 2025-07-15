import telebot
from typing import List, Tuple
from io import BytesIO


def send_messages(bot_token: str, chat_id: str, lst: List[Tuple[BytesIO, str]], functions: List[str]):
    bot = telebot.TeleBot(bot_token)

    print()
    for m, function in zip(lst, functions):
        if m is None or not (isinstance(m[0], BytesIO) or isinstance(m[1], str)):
            print(f'Incorrect data format for function "{function}" was recieved.')
            continue
        bot.send_photo(chat_id=chat_id, photo=m[0], caption=m[1], parse_mode='Markdown')
    print()
