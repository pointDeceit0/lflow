import telebot
from typing import List, Tuple, Union
from io import BytesIO


def send_messages(bot_token: str, chat_id: str,
                  lst: List[Union[Tuple[BytesIO, str], List[Tuple[BytesIO, str]]]], functions: List[str]):
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
        media = []
        # case with [[Tuple[BytesIO, str], Tuple[BytesIO, str], ...]] when multiple plots must be union in one
        if isinstance(function, list):
            for order, (subm, _) in enumerate(zip(m, function)):
                if subm is None or not (isinstance(subm[0], BytesIO) or isinstance(subm[1], str)):
                    print(f'Incorrect data format for function "{function}" was recieved.')
                    break
                media.append(
                    telebot.types.InputMediaPhoto(subm[0], caption=subm[1] if order == 0 else '', parse_mode='Markdown')
                )
        elif m is None or not (isinstance(m[0], BytesIO) or isinstance(m[1], str)):
            print(f'Incorrect data format for function "{function}" was recieved.')
            continue
        else:  # case with one plot
            media.append(
                telebot.types.InputMediaPhoto(m[0], caption=m[1], parse_mode='Markdown')
            )
        bot.send_media_group(chat_id, media=media)
    print()
