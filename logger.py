import requests
import os
import time

API_ENDPOINT = os.getenv("API_ENDPOINT")
API_TOKEN = os.getenv("API_TOKEN")

INFO = "info"
ERROR = "error"


def log(message: str, fields: dict, level=INFO) -> None:
    """Basic logger.
    :param message: short message to summarize the log statement
    :param fields: any detailed fields to send
    :param level: INFO or ERROR
    :return: None
    """
    if fields and not isinstance(fields, dict):
        raise Exception("fields must be of type dict")

    data = {}
    if fields:
        data = fields

    data["level"] = level
    data["message"] = message
    data["ts"] = time.time()

    print(data)

    _send(data)


def _send(data: dict) -> requests.Response:
    """Raw HTTP Post request to the logging API Endpoint.
    For internal use only. Don't call this func directly.
    :param data: the body of the request
    :return: requests.Response
    """
    res = None
    try:
        res = requests.post(
            url=API_ENDPOINT, data=data, headers={"Authorization": API_TOKEN}
        )
    except Exception as e:
        print("error sending log", e)
    finally:
        return res


if __name__ == "__main__":
    # Example usage
    log(
        "this is a log message",
        fields={
            "some_key": "include whatever key_values you want here.",
            "user_face": "ben",
        },
    )
