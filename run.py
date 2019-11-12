from threading import Thread
import requests
import json
import time
from compute import worker1, worker2, worker3
from parking_state import Settings

Global = Settings()
def proccessFeed():

    url = 'https://i2yv4ll3q7.execute-api.eu-west-1.amazonaws.com/hack/space/current'
    payload = json.dumps(Global.device.state)
    print(payload)
    response = requests.post(url, data=payload, allow_redirects=True)

    if response.status_code == 200:
        print("Successfully  posted")
    else:
        print("Error: Bad request")

if __name__ == '__main__':

    # Register Workers & Spawn
    Thread(target=worker1.worker1).start()
    Thread(target=worker2.worker2).start()
    Thread(target=worker3.worker3).start()

    while True:
        if len(Global.device.state) > 0:
            proccessFeed()
        time.sleep(5)