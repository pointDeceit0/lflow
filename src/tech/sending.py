import telebot
from typing import List, Tuple
from io import BytesIO


def send_messages(bot_token: str, chat_id: str, lst: List[Tuple[BytesIO, str]], functions: List[str]):
    """Sending plots with text to them in telegram

    Args:
        bot_token (str): telegram bot token
        chat_id (str): telegram chat id
        lst (List[Tuple[BytesIO, str]]): list of tuples where each one contains BytesIO view of plot firstly, and
                str message to this plot secondly
        functions (List[str]): according to lst launced functions.
    """
    bot = telebot.TeleBot(bot_token)

    print()
    for m, function in zip(lst, functions):
        if m is None or not (isinstance(m[0], BytesIO) or isinstance(m[1], str)):
            print(f'Incorrect data format for function "{function}" was recieved.')
            continue
        bot.send_photo(chat_id=chat_id, photo=m[0], caption=m[1], parse_mode='Markdown')
    print()
